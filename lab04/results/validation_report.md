# Validation report

Raw ingestion: **48,842 rows × 15 columns**. No rows dropped during validation.

OpenML names the target `class`; validation uses an unmodified-value view named `income`. The cached source and original DataFrame retain the original name. `?` is allowed in source categorical vocabularies and interpreted as missing; OpenML has already parsed these tokens as nulls.

| Check | Violations / findings |
|---|---:|
| Schema/shape | 0 |
| Logical types | 0 |
| Numeric ranges | 0 |
| Unknown categories | 0 |
| Missing target labels | 0 |
| Allowed missing predictor cells | 6465 |
| Exact duplicate additional records | 52 |
| Conflicting duplicate feature groups | 5 |

## Schema and ranges

| Field | Logical type | Source dtype | Range / category count | Violations |
|---|---|---|---|---:|
| age | int | int64 | 17–90 | 0 |
| workclass | category | category | 9 allowed categories | 0 |
| fnlwgt | int | int64 | 1–unbounded | 0 |
| education | category | category | 17 allowed categories | 0 |
| education-num | int | int64 | 1–16 | 0 |
| marital-status | category | category | 8 allowed categories | 0 |
| occupation | category | category | 15 allowed categories | 0 |
| relationship | category | category | 7 allowed categories | 0 |
| race | category | category | 6 allowed categories | 0 |
| sex | category | category | 3 allowed categories | 0 |
| capital-gain | int | int64 | 0–99999 | 0 |
| capital-loss | int | int64 | 0–4356 | 0 |
| hours-per-week | int | int64 | 1–99 | 0 |
| native-country | category | category | 42 allowed categories | 0 |
| income | category | category | 3 allowed categories | 0 |

Integer fields are checked for finite integral values. Complete allowed-category vocabularies and machine-readable checks are stored in `config_resolved.yaml`.

## Missing values

| Column | Raw nulls | Literal `?` | Total missing |
|---|---:|---:|---:|
| age | 0 | 0 | 0 |
| workclass | 2799 | 0 | 2799 |
| fnlwgt | 0 | 0 | 0 |
| education | 0 | 0 | 0 |
| education-num | 0 | 0 | 0 |
| marital-status | 0 | 0 | 0 |
| occupation | 2809 | 0 | 2809 |
| relationship | 0 | 0 | 0 |
| race | 0 | 0 | 0 |
| sex | 0 | 0 | 0 |
| capital-gain | 0 | 0 | 0 |
| capital-loss | 0 | 0 | 0 |
| hours-per-week | 0 | 0 | 0 |
| native-country | 857 | 0 | 857 |
| income | 0 | 0 | 0 |

Duplicates and allowed missing predictors are quality findings. A structural, type, range, unknown-category, or missing-target violation stops processing after the report is saved.

## Label quality

The full raw-data audit found **5 conflicting feature groups / 10 rows**. Each group has identical raw predictors and different recorded income labels. This does not establish which label is correct. No labels are reassigned.

| Conflict group | Source row ID | Income |
|---|---:|---|
| 1 | 1781 | <=50K |
| 2 | 2373 | >50K |
| 3 | 11692 | <=50K |
| 4 | 15802 | >50K |
| 3 | 16161 | >50K |
| 5 | 17129 | <=50K |
| 4 | 27210 | <=50K |
| 1 | 32964 | >50K |
| 5 | 39042 | >50K |
| 2 | 45787 | <=50K |

Full raw conflict records are saved under `label_audit.duplicate_conflicts.records` in `config_resolved.yaml`.

A seeded sample of 50 raw records (random_state=42) has 0 impossible-value or education-code inconsistencies detected automatically. Checks cover dictionary categories, integer types, ranges, missing targets and education/education-num consistency. Allowed missing predictors are reported.

**Human review: pending_user_review (0/50 completed).** All 50 source records, automated checks and reviewer/verdict/notes fields are combined in `manual_review.csv`. Automated checks do not constitute a student manual review or verify true income.

Dictionary: [UCI Adult documentation](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names). Adult uses recorded labels, so the alternative subjective-label relabeling and Cohen's kappa procedure does not apply.
