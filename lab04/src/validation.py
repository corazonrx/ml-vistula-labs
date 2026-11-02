"""Read-only validation and review artifacts for the UCI Adult data.

The public functions never clean, drop, or modify input rows. The schema describes
logical values: OpenML stores some integer-valued columns as floats and represents
UCI's ``?`` marker as a native null. Both representations are reported explicitly.
"""

from __future__ import annotations

import json
from hashlib import sha256
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


UCI_DICTIONARY_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names"
)


def _category(values: str) -> dict[str, Any]:
    return {"type": "category", "allowed": values.split("|") + ["?"]}


SCHEMA: dict[str, Any] = {
    "target": "income",
    "expected_rows": 48842,
    "missing_markers": ["?"],
    "allow_predictor_missing": True,
    "dictionary_url": UCI_DICTIONARY_URL,
    "fields": {
        "age": {"type": "int", "min": 17, "max": 90},
        "workclass": _category(
            "Private|Self-emp-not-inc|Self-emp-inc|Federal-gov|Local-gov|"
            "State-gov|Without-pay|Never-worked"
        ),
        "fnlwgt": {"type": "int", "min": 1},
        "education": _category(
            "Bachelors|Some-college|11th|HS-grad|Prof-school|Assoc-acdm|"
            "Assoc-voc|9th|7th-8th|12th|Masters|1st-4th|10th|Doctorate|"
            "5th-6th|Preschool"
        ),
        "education-num": {"type": "int", "min": 1, "max": 16},
        "marital-status": _category(
            "Married-civ-spouse|Divorced|Never-married|Separated|Widowed|"
            "Married-spouse-absent|Married-AF-spouse"
        ),
        "occupation": _category(
            "Tech-support|Craft-repair|Other-service|Sales|Exec-managerial|"
            "Prof-specialty|Handlers-cleaners|Machine-op-inspct|Adm-clerical|"
            "Farming-fishing|Transport-moving|Priv-house-serv|Protective-serv|"
            "Armed-Forces"
        ),
        "relationship": _category(
            "Wife|Own-child|Husband|Not-in-family|Other-relative|Unmarried"
        ),
        "race": _category("White|Asian-Pac-Islander|Amer-Indian-Eskimo|Other|Black"),
        "sex": _category("Female|Male"),
        "capital-gain": {"type": "int", "min": 0, "max": 99999},
        "capital-loss": {"type": "int", "min": 0, "max": 4356},
        "hours-per-week": {"type": "int", "min": 1, "max": 99},
        "native-country": _category(
            "United-States|Cambodia|England|Puerto-Rico|Canada|Germany|"
            "Outlying-US(Guam-USVI-etc)|India|Japan|Greece|South|China|Cuba|"
            "Iran|Honduras|Philippines|Italy|Poland|Jamaica|Vietnam|Mexico|"
            "Portugal|Ireland|France|Dominican-Republic|Laos|Ecuador|Taiwan|"
            "Haiti|Columbia|Hungary|Guatemala|Nicaragua|Scotland|Thailand|"
            "Yugoslavia|El-Salvador|Trinadad&Tobago|Peru|Hong|Holand-Netherlands"
        ),
        "income": _category("<=50K|>50K"),
    },
}

EDUCATION_NUMBERS = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4,
    "9th": 5, "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9,
    "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
    "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16,
}


def _target(schema: dict[str, Any]) -> str:
    return schema.get("target_name", schema.get("target", "income"))


def _fields(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Support a configurable target name without modifying the shared schema."""
    fields = dict(schema.get("fields", SCHEMA["fields"]))
    target = _target(schema)
    if target != "income" and "income" in fields and target not in fields:
        fields[target] = fields.pop("income")
    return fields


def _safe(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, tuple):
        return [_safe(item) for item in value]
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _missing(series: pd.Series, schema: dict[str, Any]) -> pd.Series:
    return series.isna() | series.isin(schema.get("missing_markers", ["?"]))


def _examples(series: pd.Series, mask: pd.Series, limit: int = 5) -> list[dict]:
    return [
        {"source_row_id": _safe(index), "value": _safe(value)}
        for index, value in series[mask].iloc[:limit].items()
    ]


def _integer_mask(series: pd.Series) -> pd.Series:
    def is_integer(value: Any) -> bool:
        return (
            isinstance(value, Real)
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value)
            and float(value).is_integer()
        )
    return series.map(is_integer).astype(bool)


def _normalized(frame: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    # Object dtype avoids categorical groupby dropping missing-value groups.
    values = frame.astype(object).copy()
    for column in values:
        values[column] = values[column].mask(_missing(values[column], schema), np.nan)
    return values


def _conflicts(
    frame: pd.DataFrame, schema: dict[str, Any]
) -> tuple[pd.DataFrame, int]:
    target = _target(schema)
    features = [column for column in _fields(schema) if column != target]
    if target not in frame or any(column not in frame for column in features):
        return pd.DataFrame(columns=["conflict_group_id", "source_row_id", *frame.columns]), 0
    view = _normalized(frame[features + [target]], schema)
    grouped = view.groupby(features, dropna=False, sort=False, observed=True)
    mask = grouped[target].transform("nunique").gt(1)
    group_ids = grouped.ngroup()[mask]
    id_map = {identifier: i + 1 for i, identifier in enumerate(pd.unique(group_ids))}
    rows = frame.loc[mask].copy()
    rows.insert(0, "source_row_id", [_safe(index) for index in rows.index])
    rows.insert(0, "conflict_group_id", group_ids.map(id_map).to_numpy())
    return rows.reset_index(drop=True), len(id_map)


def validate_frame(frame: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any]:
    """Return JSON-safe checks; permit missing predictors but never missing labels.

    ``schema`` normally is ``SCHEMA``. Override ``target``/``target_name`` for an
    unrenamed OpenML frame. Omit ``expected_rows`` when validating a small fixture.
    """
    fields = _fields(schema)
    target = _target(schema)
    missing_columns = [column for column in fields if column not in frame]
    unexpected_columns = [column for column in frame if column not in fields]
    expected_rows = schema.get("expected_rows")
    row_count_matches = expected_rows is None or len(frame) == expected_rows
    report: dict[str, Any] = {
        "shape": {"rows": len(frame), "columns": len(frame.columns),
                  "expected_rows": expected_rows, "expected_columns": len(fields),
                  "row_count_matches": row_count_matches},
        "schema": {"missing_columns": missing_columns,
                   "unexpected_columns": unexpected_columns,
                   "logical_types": {name: spec["type"] for name, spec in fields.items()},
                   "storage_dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()}},
        "missing": {}, "types": {}, "ranges": {}, "categories": {},
    }
    type_count = range_count = category_count = predictor_missing = target_missing = 0
    for column, spec in fields.items():
        if column not in frame:
            continue
        series = frame[column]
        nulls = series.isna()
        markers = series.isin(schema.get("missing_markers", ["?"]))
        missing = nulls | markers
        count = int(missing.sum())
        report["missing"][column] = {
            "native_null_count": int(nulls.sum()), "question_mark_count": int(markers.sum()),
            "total_missing": count, "allowed": column != target and schema.get("allow_predictor_missing", True),
            "examples": _examples(series, missing),
        }
        if column == target:
            target_missing += count
        else:
            predictor_missing += count
        if spec["type"] == "int":
            valid_type = _integer_mask(series)
        else:
            valid_type = series.map(lambda value: isinstance(value, str)).astype(bool)
        type_bad = ~missing & ~valid_type
        type_count += int(type_bad.sum())
        report["types"][column] = {
            "expected": spec["type"], "storage_dtype": str(series.dtype),
            "violations": int(type_bad.sum()), "examples": _examples(series, type_bad),
        }
        if spec["type"] == "int":
            numeric = pd.to_numeric(series.astype(object), errors="coerce")
            range_bad = pd.Series(False, index=series.index)
            if "min" in spec:
                range_bad |= numeric.lt(spec["min"])
            if "max" in spec:
                range_bad |= numeric.gt(spec["max"])
            range_bad &= ~missing & valid_type
            range_count += int(range_bad.sum())
            report["ranges"][column] = {
                "min": spec.get("min"), "max": spec.get("max"),
                "violations": int(range_bad.sum()), "examples": _examples(series, range_bad),
            }
        else:
            category_bad = ~missing & ~series.isin(spec["allowed"])
            category_count += int(category_bad.sum())
            report["categories"][column] = {
                "allowed": spec["allowed"], "violations": int(category_bad.sum()),
                "examples": _examples(series, category_bad),
            }
    normalized = _normalized(frame, schema)
    exact_all = normalized.duplicated(keep=False)
    exact_extra = normalized.duplicated(keep="first")
    conflict_rows, conflict_groups = _conflicts(frame, schema)
    report["duplicates"] = {
        "exact_duplicate_rows_including_first": int(exact_all.sum()),
        "exact_duplicate_additional_rows": int(exact_extra.sum()),
        "exact_duplicate_groups": int(normalized.loc[exact_all].drop_duplicates().shape[0]),
        "conflicting_feature_groups": conflict_groups,
        "rows_in_conflicting_feature_groups": len(conflict_rows),
        "policy": "Report only. Exact duplicate removal is allowed on training rows after splitting; test rows are retained.",
    }
    disallowed_missing = target_missing
    if not schema.get("allow_predictor_missing", True):
        disallowed_missing += predictor_missing
    structural_count = len(missing_columns) + len(unexpected_columns) + int(not row_count_matches)
    report["summary"] = {
        "structural_violations": structural_count,
        "type_violation_cells": type_count, "range_violation_cells": range_count,
        "category_violation_cells": category_count,
        "missing_predictor_cells": predictor_missing, "missing_target_rows": target_missing,
        "disallowed_missing_cells": disallowed_missing,
        "total_violations": structural_count + type_count + range_count + category_count + disallowed_missing,
        "rows_dropped": 0,
    }
    report["fatal"] = report["summary"]["total_violations"] > 0
    # Fail here if a future change accidentally introduces a non-serializable value.
    json.dumps(report, allow_nan=False)
    return report


def _row_issues(row: pd.Series, schema: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    target = _target(schema)
    for column, spec in _fields(schema).items():
        if column not in row:
            issues.append(f"{column}: missing column")
            continue
        value = row[column]
        missing = pd.isna(value) or value in schema.get("missing_markers", ["?"])
        if missing:
            if column == target:
                issues.append(f"{column}: missing label")
            continue
        if spec["type"] == "int":
            if not _integer_mask(pd.Series([value])).iloc[0]:
                issues.append(f"{column}: not an integer")
            elif ("min" in spec and value < spec["min"]) or ("max" in spec and value > spec["max"]):
                issues.append(f"{column}: outside dictionary/lab range")
        elif value not in spec["allowed"]:
            issues.append(f"{column}: unknown category")
    education = row.get("education")
    number = row.get("education-num")
    if isinstance(education, str) and education in EDUCATION_NUMBERS and not pd.isna(number):
        if number != EDUCATION_NUMBERS[education]:
            issues.append("education and education-num disagree")
    return issues


def label_audit(
    frame: pd.DataFrame,
    schema: dict[str, Any],
    seed: int,
    sample_size: int,
    results_dir: Path,
) -> dict[str, Any]:
    """Return full conflict evidence and save one human-review worksheet.

    The dictionary check is automated; human sign-off remains blank and pending.
    Duplicate conflicts establish inconsistent recorded labels, not which is true.
    """
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    conflict_rows, groups = _conflicts(frame, schema)
    sample = frame.sample(n=min(sample_size, len(frame)), random_state=seed).copy()
    sample.insert(0, "source_row_id", [_safe(index) for index in sample.index])
    checks = []
    for source_index, row in sample.iterrows():
        issues = _row_issues(row, schema)
        checks.append({
            "source_row_id": _safe(source_index),
            "dictionary_check_passed": not issues,
            "impossible_value_count": len(issues),
            "issues": "; ".join(issues),
            "missing_predictor_count": sum(
                pd.isna(row[column]) or row[column] in schema.get("missing_markers", ["?"])
                for column in _fields(schema) if column != _target(schema) and column in row
            ),
        })
    checks_frame = pd.DataFrame(checks, columns=[
        "source_row_id", "dictionary_check_passed", "impossible_value_count", "issues", "missing_predictor_count"
    ])
    review = sample.reset_index(drop=True).copy()
    review["automated_dictionary_check"] = checks_frame["dictionary_check_passed"].map({True: "pass", False: "fail"})
    review["automated_issues"] = checks_frame["issues"]
    review["automated_impossible_value_count"] = checks_frame["impossible_value_count"]
    review["missing_predictor_count"] = checks_frame["missing_predictor_count"]
    review_path = results_dir / "manual_review.csv"
    previous_review = None
    review_archive = None
    if review_path.exists():
        previous_review = pd.read_csv(
            review_path, dtype={"source_row_id": str}, keep_default_na=False
        )
        if "source_row_id" not in previous_review:
            raise ValueError("Existing manual_review.csv has no source_row_id; refusing to overwrite it")
        if previous_review["source_row_id"].duplicated().any():
            raise ValueError("Existing manual_review.csv has duplicate source_row_id values; refusing to overwrite it")
        previous_review = previous_review.set_index("source_row_id")
        # A new seed can select different rows. Keep any displaced human work
        # in a content-addressed archive instead of silently discarding it.
        displaced = previous_review.loc[
            ~previous_review.index.isin(review["source_row_id"].astype(str))
        ]
        review_fields = [column for column in ("reviewer", "verdict", "notes") if column in displaced]
        if review_fields and displaced[review_fields].astype(str).ne("").any().any():
            prior_bytes = review_path.read_bytes()
            review_archive = f"manual_review_archive_{sha256(prior_bytes).hexdigest()[:12]}.csv"
            (results_dir / review_archive).write_bytes(prior_bytes)
    for column in ("reviewer", "verdict", "notes"):
        review[column] = ""
        if previous_review is not None and column in previous_review:
            review[column] = review["source_row_id"].astype(str).map(previous_review[column]).fillna("")
    review.to_csv(review_path, index=False)
    signed_off = (
        review["reviewer"].astype(str).str.strip().ne("")
        & review["verdict"].astype(str).str.strip().str.lower().isin(["valid", "impossible", "uncertain"])
    )
    completed_rows = int(signed_off.sum())
    report = {
        "duplicate_conflicts": {
            "groups": groups, "rows": len(conflict_rows),
            "records": [
                {column: _safe(value) for column, value in row.items()}
                for row in conflict_rows.to_dict(orient="records")
            ],
            "labels_are_not_reassigned": True,
            "definition": "Identical raw predictor values, treating '?' and native null as the same missing value, with more than one nonmissing income label.",
        },
        "sample": {
            "seed": int(seed), "requested_rows": int(sample_size), "sampled_rows": len(sample),
            "source_row_ids": [_safe(index) for index in sample.index],
            "automated_dictionary_passed_rows": int(checks_frame["dictionary_check_passed"].sum()),
            "automated_dictionary_failed_rows": int((~checks_frame["dictionary_check_passed"].astype(bool)).sum()),
            "impossible_value_count": int(checks_frame["impossible_value_count"].sum()),
            "dictionary_url": schema.get("dictionary_url", UCI_DICTIONARY_URL),
            "automated_checks": "Logical integer types, fixed category vocabulary, lab ranges, missing target, and education/education-num consistency. Missing predictors are reported and allowed.",
        },
        "manual_review": {
            "status": "complete" if len(review) > 0 and completed_rows == len(review) else "pending_user_review",
            "completed_rows": completed_rows,
            "previous_reviewer_fields_preserved": previous_review is not None,
            "displaced_review_archive": review_archive,
            "instructions": "Inspect each sampled row against the UCI data dictionary. Fill reviewer, verdict (valid/impossible/uncertain), and notes in manual_review.csv. Automated checks do not constitute a human review or verification of the income label.",
        },
        "artifacts": {
            "manual_review": "manual_review.csv",
        },
    }
    json.dumps(report, allow_nan=False)
    return report
