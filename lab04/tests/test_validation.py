"""Inject data-quality faults and check read-only audit evidence."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validation import SCHEMA, label_audit, validate_frame


@pytest.fixture
def schema() -> dict:
    specification = copy.deepcopy(SCHEMA)
    specification.pop("expected_rows")  # These deliberately small fixtures are not ingestion.
    return specification


def valid_rows(count: int = 6) -> pd.DataFrame:
    values = {
        "age": 30.0,
        "workclass": "Private",
        "fnlwgt": 150000,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Tech-support",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
        "income": "<=50K",
    }
    frame = pd.DataFrame([values.copy() for _ in range(count)], index=range(1001, 1001 + count))
    for column, specification in SCHEMA["fields"].items():
        if specification["type"] == "category":
            frame[column] = pd.Categorical(frame[column], categories=specification["allowed"])
    return frame


def test_predictor_missing_markers_and_native_nulls_are_allowed(schema):
    frame = valid_rows(3)
    frame.loc[1001, "workclass"] = "?"
    frame.loc[1002, "workclass"] = np.nan
    frame.loc[1003, "age"] = np.nan
    original = frame.copy(deep=True)

    report = validate_frame(frame, schema)

    assert report["fatal"] is False
    assert report["summary"]["total_violations"] == 0
    assert report["summary"]["missing_predictor_cells"] == 3
    assert report["missing"]["workclass"]["question_mark_count"] == 1
    assert report["missing"]["workclass"]["native_null_count"] == 1
    assert report["missing"]["age"]["native_null_count"] == 1
    assert report["types"]["age"]["violations"] == 0  # 30.0 is logically an integer.
    assert report["schema"]["storage_dtypes"]["age"] == "float64"
    assert report["schema"]["storage_dtypes"]["workclass"] == "category"
    json.dumps(report, allow_nan=False)
    assert_frame_equal(frame, original)


@pytest.mark.parametrize("missing_label", [np.nan, "?"])
def test_faults_are_fatal_report_source_rows_and_preserve_input(schema, missing_label):
    frame = valid_rows(4)
    frame["workclass"] = frame["workclass"].cat.add_categories(["Unlisted-employer"])
    frame.loc[1001, "age"] = 30.5
    frame.loc[1002, "hours-per-week"] = 100
    frame.loc[1003, "workclass"] = "Unlisted-employer"
    frame.loc[1004, "income"] = missing_label
    original = frame.copy(deep=True)

    report = validate_frame(frame, schema)

    assert report["fatal"] is True
    assert report["summary"]["total_violations"] == 4
    assert report["summary"]["rows_dropped"] == 0
    assert report["types"]["age"]["violations"] == 1
    assert report["types"]["age"]["examples"] == [{"source_row_id": 1001, "value": 30.5}]
    assert report["ranges"]["hours-per-week"]["violations"] == 1
    assert report["ranges"]["hours-per-week"]["examples"] == [{"source_row_id": 1002, "value": 100}]
    assert report["categories"]["workclass"]["violations"] == 1
    assert report["categories"]["workclass"]["examples"] == [
        {"source_row_id": 1003, "value": "Unlisted-employer"}
    ]
    assert report["summary"]["missing_target_rows"] == 1
    assert report["missing"]["income"]["allowed"] is False
    assert report["missing"]["income"]["examples"][0]["source_row_id"] == 1004
    json.dumps(report, allow_nan=False)
    assert_frame_equal(frame, original)


def test_conflict_audit_groups_native_null_and_question_mark_without_cleaning(schema, tmp_path):
    frame = valid_rows(7)
    frame.loc[[1001, 1002, 1007], "workclass"] = ["?", np.nan, "?"]
    frame.loc[1002, "income"] = ">50K"
    frame.loc[[1003, 1004], "age"] = 40
    frame.loc[[1005, 1006], "age"] = 50
    frame.loc[[1005, 1006], "income"] = ">50K"
    original = frame.copy(deep=True)

    validation = validate_frame(frame, schema)
    audit = label_audit(frame, schema, seed=42, sample_size=50, results_dir=tmp_path)
    conflicts = pd.DataFrame(audit["duplicate_conflicts"]["records"])
    worksheet = pd.read_csv(tmp_path / "manual_review.csv", keep_default_na=False)

    assert validation["duplicates"]["exact_duplicate_additional_rows"] == 3
    assert validation["duplicates"]["conflicting_feature_groups"] == 1
    assert audit["duplicate_conflicts"]["groups"] == 1
    assert audit["duplicate_conflicts"]["rows"] == 3
    assert set(conflicts["source_row_id"]) == {1001, 1002, 1007}
    assert conflicts["conflict_group_id"].nunique() == 1
    assert set(conflicts["income"]) == {"<=50K", ">50K"}
    assert "?" in conflicts["workclass"].values
    assert conflicts["workclass"].isna().sum() == 1  # Evidence keeps raw representations.
    assert set(conflicts.columns) == {"conflict_group_id", "source_row_id", *frame.columns}
    assert len(worksheet) == 7 and set(worksheet["source_row_id"]) == set(frame.index)
    assert worksheet["automated_dictionary_check"].eq("pass").all()
    assert worksheet["automated_impossible_value_count"].eq(0).all()
    assert worksheet["missing_predictor_count"].sum() == 3
    assert worksheet[["reviewer", "verdict", "notes"]].eq("").all().all()
    assert audit["manual_review"]["status"] == "pending_user_review"
    assert audit["manual_review"]["completed_rows"] == 0
    assert audit["artifacts"] == {"manual_review": "manual_review.csv"}
    assert {path.name for path in tmp_path.iterdir()} == {"manual_review.csv"}
    json.dumps(audit, allow_nan=False)
    assert_frame_equal(frame, original)


def test_manual_reviewer_fields_survive_reproducible_rerun(schema, tmp_path):
    frame = valid_rows(6)
    first = label_audit(frame, schema, seed=42, sample_size=3, results_dir=tmp_path)
    path = tmp_path / "manual_review.csv"
    worksheet = pd.read_csv(path, keep_default_na=False)
    reviewed_id = worksheet.loc[0, "source_row_id"]
    worksheet.loc[0, ["reviewer", "verdict", "notes"]] = [
        "Student reviewer", "valid", "Compared all fields with the UCI dictionary."
    ]
    worksheet.to_csv(path, index=False)

    repeated = label_audit(frame, schema, seed=42, sample_size=3, results_dir=tmp_path)
    persisted = pd.read_csv(path, keep_default_na=False).set_index("source_row_id")

    assert repeated["sample"]["source_row_ids"] == first["sample"]["source_row_ids"]
    assert persisted.loc[reviewed_id, "reviewer"] == "Student reviewer"
    assert persisted.loc[reviewed_id, "verdict"] == "valid"
    assert persisted.loc[reviewed_id, "notes"] == "Compared all fields with the UCI dictionary."
    assert repeated["manual_review"]["completed_rows"] == 1
    assert repeated["manual_review"]["status"] == "pending_user_review"


def test_changed_sample_archives_displaced_human_review(schema, tmp_path):
    frame = valid_rows(6)
    label_audit(frame, schema, seed=42, sample_size=2, results_dir=tmp_path)
    path = tmp_path / "manual_review.csv"
    worksheet = pd.read_csv(path, keep_default_na=False)
    worksheet.loc[:, ["reviewer", "verdict", "notes"]] = ["Reviewer", "valid", "Reviewed"]
    worksheet.to_csv(path, index=False)
    reviewed_bytes = path.read_bytes()

    changed = label_audit(frame, schema, seed=1, sample_size=1, results_dir=tmp_path)

    archive = changed["manual_review"]["displaced_review_archive"]
    assert archive is not None
    assert (tmp_path / archive).read_bytes() == reviewed_bytes
