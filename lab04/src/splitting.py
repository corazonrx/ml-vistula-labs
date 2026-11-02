"""Split raw observations without sharing repeated feature profiles.

Stratification alone can put duplicate profiles on opposite sides of a split.
This module keeps every identical raw feature profile together, including
profiles whose target labels disagree, while retaining the exact class totals
of scikit-learn's ordinary stratified 70/30 split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_group_split(
    frame: pd.DataFrame,
    target: str = "income",
    test_size: float = 0.3,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return untouched raw partitions with exact stratification and no overlap.

    Multirow feature groups are allocated as units using a seeded generator.
    Singleton groups then fill each class's remaining test quota. If duplicate
    groups make the quota infeasible, allocation is retried deterministically
    up to 100 times; rows are never discarded to obtain the requested split.
    """
    if target not in frame:
        raise ValueError(f"Target column {target!r} is missing.")
    if len(frame.columns) < 2:
        raise ValueError("A grouped split requires at least one feature column.")
    if not frame.index.is_unique:
        raise ValueError("Raw source row indices must be unique before splitting.")
    if frame[target].isna().any():
        raise ValueError("Target labels contain missing values; split is undefined.")
    if not isinstance(test_size, (float, np.floating)) or not 0 < test_size < 1:
        raise ValueError("test_size must be a fraction strictly between 0 and 1.")

    positions = np.arange(len(frame))
    try:
        _, ordinary_test = train_test_split(
            positions,
            test_size=test_size,
            stratify=frame[target],
            random_state=seed,
        )
    except ValueError as error:
        raise ValueError(f"Cannot create a stratified split: {error}") from error

    # Convert only the grouping view. Raw values, dtypes, and source indices
    # remain unchanged in frame and in the returned partitions. Object dtypes
    # make dropna=False consistent for categorical columns with missing values.
    features = frame.drop(columns=[target]).astype(object)
    group_codes = features.groupby(
        list(features.columns), dropna=False, observed=True, sort=False
    ).ngroup().to_numpy(dtype=np.int64)
    group_sizes = np.bincount(group_codes)
    duplicate_groups = np.flatnonzero(group_sizes > 1)

    class_codes, classes = pd.factorize(frame[target], sort=False)
    n_classes = len(classes)
    desired_test = np.bincount(class_codes[ordinary_test], minlength=n_classes)
    singleton_positions = [
        np.flatnonzero((group_sizes[group_codes] == 1) & (class_codes == code))
        for code in range(n_classes)
    ]
    singleton_available = np.array([len(rows) for rows in singleton_positions])

    # One row per duplicate group, one column per target class. A conflicting
    # profile contributes to multiple columns and is still allocated whole.
    duplicate_counts = np.zeros((len(duplicate_groups), n_classes), dtype=int)
    duplicate_group_lookup = np.full(len(group_sizes), -1, dtype=int)
    duplicate_group_lookup[duplicate_groups] = np.arange(len(duplicate_groups))
    repeated_rows = group_sizes[group_codes] > 1
    np.add.at(
        duplicate_counts,
        (duplicate_group_lookup[group_codes[repeated_rows]], class_codes[repeated_rows]),
        1,
    )

    rng = np.random.default_rng(seed)
    test_mask = None
    for _ in range(100):
        chosen_duplicates = rng.random(len(duplicate_groups)) < test_size
        needed = desired_test - duplicate_counts[chosen_duplicates].sum(axis=0)
        if np.any(needed < 0) or np.any(needed > singleton_available):
            continue
        selected_groups = duplicate_groups[chosen_duplicates]
        candidate_mask = np.isin(group_codes, selected_groups)
        for code, count in enumerate(needed):
            selected_rows = rng.choice(
                singleton_positions[code], size=int(count), replace=False
            )
            candidate_mask[selected_rows] = True
        test_mask = candidate_mask
        break

    if test_mask is None:
        raise ValueError(
            "Cannot preserve both exact stratification and whole duplicate feature "
            "groups after 100 seeded attempts; there may be too few singleton "
            "profiles for the requested split. No rows were dropped."
        )

    train = frame.iloc[np.flatnonzero(~test_mask)].sort_index().copy()
    test = frame.iloc[np.flatnonzero(test_mask)].sort_index().copy()
    crossing_groups = np.intersect1d(
        np.unique(group_codes[~test_mask]), np.unique(group_codes[test_mask])
    )
    if len(crossing_groups) or not np.array_equal(
        np.bincount(class_codes[test_mask], minlength=n_classes), desired_test
    ):
        raise RuntimeError("Internal split invariant failed.")

    def class_counts(partition: pd.DataFrame) -> dict[str, int]:
        counts = partition[target].value_counts(sort=False)
        return {str(label): int(count) for label, count in counts.items()}

    info = {
        "strategy": "seeded_whole_feature_groups_then_exact_stratified_singletons",
        "train_rows_before_dedup": len(train),
        "test_rows": len(test),
        "train_class_counts": class_counts(train),
        "test_class_counts": class_counts(test),
        "duplicate_feature_groups": int(len(duplicate_groups)),
        "feature_groups_crossing_split": int(len(crossing_groups)),
    }
    return train, test, info


def deduplicate_training(train: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove only complete repeated training records, retaining source indices."""
    duplicate_mask = train.duplicated(keep="first")
    return train.loc[~duplicate_mask].copy(), int(duplicate_mask.sum())
