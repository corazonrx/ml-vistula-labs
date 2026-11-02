"""Focused checks of leakage boundaries and raw-data preservation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.splitting import deduplicate_training, stratified_group_split
from src.preprocessing import CATEGORICAL, NUMERIC, make_preprocessor


def raw_rows() -> pd.DataFrame:
    """Many singletons plus ordinary, conflicting, and missing-value repeats."""
    frame = pd.DataFrame(
        {
            "age": np.arange(20, 80),
            "workclass": pd.Categorical(["Private", "?"] * 30),
            "income": pd.Categorical(["<=50K", ">50K"] * 30),
        },
        index=pd.Index(np.arange(100, 160), name="source_row"),
    )
    extras = frame.iloc[[0, 1, 2, 3]].copy()
    extras.index = pd.Index(np.arange(160, 164), name="source_row")
    extras.loc[162, "income"] = ">50K"  # Same features, conflicting label.
    frame.loc[103, "workclass"] = np.nan
    extras.loc[163, "workclass"] = np.nan
    return pd.concat([frame, extras])


def feature_hashes(frame: pd.DataFrame) -> set[int]:
    return set(
        pd.util.hash_pandas_object(frame.drop(columns="income"), index=False).tolist()
    )


def test_grouped_split_has_exact_stratification_and_no_feature_leakage():
    frame = raw_rows()
    train, test, info = stratified_group_split(frame)
    _, ordinary_test = train_test_split(
        frame, test_size=0.3, random_state=42, stratify=frame["income"]
    )

    assert len(train) + len(test) == len(frame)
    assert set(train.index).isdisjoint(test.index)
    assert set(train.index) | set(test.index) == set(frame.index)
    assert feature_hashes(train).isdisjoint(feature_hashes(test))
    assert test["income"].value_counts().to_dict() == ordinary_test["income"].value_counts().to_dict()
    assert info["duplicate_feature_groups"] == 4
    assert info["feature_groups_crossing_split"] == 0
    assert info["train_rows_before_dedup"] == len(train)
    assert info["test_rows"] == len(test)
    for left, right in [(100, 160), (101, 161), (102, 162), (103, 163)]:
        assert (left in test.index) == (right in test.index)


def test_split_is_reproducible_and_preserves_raw_values_dtypes_and_indices():
    frame = raw_rows().iloc[::-1]
    original = frame.copy(deep=True)
    train, test, info = stratified_group_split(frame, seed=21)
    repeated_train, repeated_test, repeated_info = stratified_group_split(frame, seed=21)

    assert_frame_equal(frame, original)
    assert_frame_equal(train, original.loc[train.index])
    assert_frame_equal(test, original.loc[test.index])
    assert_frame_equal(train, repeated_train)
    assert_frame_equal(test, repeated_test)
    assert info == repeated_info
    assert train.index.is_monotonic_increasing
    assert test.index.is_monotonic_increasing


def test_training_dedup_keeps_conflicting_labels_and_does_not_mutate_test():
    frame = raw_rows()
    untouched = frame.copy(deep=True)
    cleaned, removed = deduplicate_training(frame)

    assert removed == 3  # The conflicting feature profile is not an exact row duplicate.
    assert 102 in cleaned.index and 162 in cleaned.index
    assert not cleaned.duplicated().any()
    assert_frame_equal(frame, untouched)


def test_impossible_group_split_fails_without_discarding_rows():
    # Every class has a single repeated profile; each profile must remain whole.
    frame = pd.DataFrame(
        {"age": [20] * 10 + [40] * 10, "income": ["<=50K"] * 10 + [">50K"] * 10}
    )
    original = frame.copy(deep=True)
    with pytest.raises(ValueError, match="whole duplicate feature groups.*No rows were dropped"):
        stratified_group_split(frame)
    assert_frame_equal(frame, original)


def test_preprocessing_learns_only_training_statistics_and_preserves_raw_test():
    train = pd.DataFrame(
        {
            "age": [20, 30, np.nan, 40, 50],
            "fnlwgt": [100, 200, np.nan, 300, 400],
            "education-num": [1, 2, np.nan, 3, 4],
            "capital-gain": [0, 10, np.nan, 20, 30],
            "capital-loss": [0, 1, np.nan, 2, 3],
            "hours-per-week": [10, 20, np.nan, 30, 40],
            **{
                column: pd.Categorical(["a", "a", "b", "?", "a"])
                for column in CATEGORICAL
            },
        }
    )
    test = train.iloc[:2].copy()
    test.loc[test.index[0], NUMERIC] = np.nan
    test.loc[test.index[1], NUMERIC] = 1e9
    for column in CATEGORICAL:
        test[column] = pd.Series(["?", "unseen"], index=test.index)
    original_train, original_test = train.copy(deep=True), test.copy(deep=True)

    transformer = make_preprocessor(engineered=True)
    transformer.fit(train)
    imputer = transformer.named_steps["impute"]
    clipper = transformer.named_steps["clip"]
    expected_medians = train[NUMERIC].median()
    expected_imputed = train[NUMERIC].fillna(expected_medians)
    np.testing.assert_allclose(imputer.numeric_.statistics_, expected_medians)
    assert list(imputer.categorical_.statistics_) == ["a"] * len(CATEGORICAL)
    pd.testing.assert_series_equal(clipper.lower_bounds_, expected_imputed.quantile(0.01))
    pd.testing.assert_series_equal(clipper.upper_bounds_, expected_imputed.quantile(0.99))
    fitted_medians = imputer.numeric_.statistics_.copy()
    fitted_upper = clipper.upper_bounds_.copy()

    encoded = transformer.transform(test)
    processed = transformer[:-1].transform(test)
    assert encoded.shape[0] == len(test)
    assert np.isfinite(encoded.data).all()
    assert processed.loc[test.index[0], "age"] == 35
    assert processed.loc[test.index[1], "age"] == pytest.approx(49.6)
    assert processed.loc[test.index[0], "workclass"] == "a"
    np.testing.assert_allclose(
        processed["capital_net"], processed["capital-gain"] - processed["capital-loss"]
    )
    np.testing.assert_allclose(
        processed["hours_per_age"], processed["hours-per-week"] / (processed["age"] - 16)
    )
    for categories in transformer.named_steps["encode"].named_transformers_["categorical"].categories_:
        assert "unseen" not in categories
        assert "?" not in categories
    np.testing.assert_array_equal(imputer.numeric_.statistics_, fitted_medians)
    pd.testing.assert_series_equal(clipper.upper_bounds_, fitted_upper)
    assert_frame_equal(train, original_train)
    assert_frame_equal(test, original_test)
