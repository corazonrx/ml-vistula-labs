# Lab 4: Data pipeline and quality

## 1. Goal

Build a reproducible data pipeline for mixed numerical and categorical inputs, validate data quality, prevent information leakage, and evaluate whether derived features improve an income classifier. The supervised binary-classification task predicts whether a person's recorded annual income exceeds US$50,000. The two outcome classes are `<=50K` and `>50K`, with `>50K` treated as the positive class. OpenML's target name `class` is represented as `income` in the lab's reports.

The 14 inputs are age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week and native-country. They combine six numerical and eight categorical variables. A row is an observation, not a guaranteed unique person. The census weighting field fnlwgt is used as an input rather than an evaluation weight. The exercise studies data engineering and predictive performance on historical records; it does not establish suitability for decisions about individuals.

Success is evaluated using accuracy and positive-class F1 on a fixed, stratified 30% test partition. Accuracy measures overall correctness, while F1 combines precision and recall for the less common higher-income class. LogisticRegression provides a baseline, compared with a predefined variant that adds net capital gain and weekly hours divided by age minus 16. Both variants use the same split and model settings. Their test results describe this experiment and are not used to tune the features, preprocessing or classifier.

The main constraints are reproducibility and prevention of information leakage. Splitting precedes cleaning, identical raw feature profiles stay together, and every test record is retained. Imputation, clipping, scaling and encoding learn only from training data. Validation must report missing values, duplicates, invalid values and label conflicts without silently deleting records. The saved transformer, configuration, processed-data SHA256, validation evidence and data card make the pipeline auditable and runnable with one command. Automated sample checks support, but do not replace, the student's manual inspection.

## 2. Method

The Adult dataset is downloaded using `fetch_openml(name='adult', version=2, as_frame=True)`, OpenML data ID 1590. It contains 48,842 records with 14 predictors (six numerical and eight categorical) and one income target, giving 15 columns. OpenML names the target `class`; reports use `income`. Ingestion preserves the original values and cached source. The positive label is `>50K`.

Validation checks column names, logical types, category vocabularies, missing values, exact duplicates and conflicting labels. Required ranges are age 17–90, hours-per-week 1–99, capital-gain 0–99999 and capital-loss 0–4356. Additional checks require positive fnlwgt and education-num 1–16. The source token `?` means missing and is allowed in categorical vocabularies; OpenML represents these values as nulls. Validation reports findings without dropping rows; disallowed violations stop the pipeline.

A stratified 70/30 split with random_state=42 is made before cleaning. Identical raw 14-feature profiles are assigned together, including profiles with conflicting labels, while preserving exact stratified class-count targets. This prevents duplicate-profile overlap between training and test. Before deduplication, training class counts are `{'<=50K': 26008, '>50K': 8181}` and test counts are `{'<=50K': 11147, '>50K': 3506}`. Exact complete duplicates are then removed from training only; every test row is retained.

Numerical median and categorical mode imputers are fitted on deduplicated training data. All six original numerical features are then clipped to their training 1st/99th percentiles. StandardScaler and OneHotEncoder with `handle_unknown='ignore'` also fit on training only. Test features receive the fitted transformations; the original test rows, labels, order and count remain unchanged. Retained row assignments are included in the processed CSV; removed training source IDs and split summaries are saved in [results/config_resolved.yaml](results/config_resolved.yaml).

Two LogisticRegression pipelines use the same split and settings: solver=lbfgs, C=1.0, max_iter=2000, tol=0.0001, random_state=42 and decision threshold=0.5. The baseline uses the original predictors. After imputation and clipping, the engineered variant adds `capital_net = capital-gain - capital-loss` and `hours_per_age = hours-per-week / (age - 16)`. The age constraint keeps the denominator positive. Neither model settings nor feature choices are tuned using test results.

Label quality is checked by finding identical raw feature profiles with different income labels and drawing 50 random records with random_state=42. Sample checks cover dictionary categories, integer types, ranges and agreement between education and education-num. A separate worksheet supports the student's manual inspection.

Run the complete pipeline from the repository root with `./lab04/run.sh`. Settings are stored in [config.yaml](config.yaml), and the effective configuration, schema, environment versions and processed-data hash are saved in [results/config_resolved.yaml](results/config_resolved.yaml). The fitted engineered transformer is saved as [results/preprocessor.joblib](results/preprocessor.joblib). Model serialization is checked in memory without saving separate model files; the transformer is reloaded from disk. Reloaded transformations and predictions match their original equivalents.

## 3. Results

| Model variant | Test accuracy | F1 (>50K) | Precision (>50K) | Recall (>50K) |
|---|---:|---:|---:|---:|
| baseline | 0.8514 | 0.6517 | 0.7416 | 0.5813 |
| engineered | 0.8542 | 0.6610 | 0.7447 | 0.5941 |

Tabular metrics are saved in [results/metrics.csv](results/metrics.csv); full-precision metrics and confusion matrices are included in [results/config_resolved.yaml](results/config_resolved.yaml). Derived features change accuracy by +0.280 percentage points and F1 by +0.009212. These are results from one predefined holdout; statistical significance has not been established. Retained test duplicates make this a record-level evaluation. No subgroup fairness claim is made.

### Data quality and partition sizes

| Check | Result |
|---|---:|
| Disallowed schema, type, range, category or required-label violations | 0 |
| Missing predictor cells | 6,465 |
| Missing workclass / occupation / native-country cells | 2,799 / 2,809 / 857 |
| Additional exact duplicate records in the full raw dataset | 52 |
| Exact duplicate training rows removed | 37 |
| Raw feature groups shared by training and test | 0 |
| Conflicting feature groups / affected rows | 5 / 10 |
| Sampled records / impossible values detected automatically | 50 / 0 |

| Partition | Before training deduplication | Retained for modelling |
|---|---:|---:|
| Training | 34,189 | 34,152 |
| Test | 14,653 | 14,653 |

Detailed validation and label-quality findings are combined in [results/validation_report.md](results/validation_report.md). Full raw conflicting records are preserved in [results/config_resolved.yaml](results/config_resolved.yaml). Conflicting profiles establish recorded-label ambiguity; they do not identify which income label is correct. No labels are reassigned. **Human-review worksheet status: pending_user_review (0/50 completed).** The 50 source records, automated row checks and human worksheet are combined in [results/manual_review.csv](results/manual_review.csv).

### Reproducibility evidence

[results/config_resolved.yaml](results/config_resolved.yaml) consolidates source integrity, preserved test rows, zero shared raw feature groups, training-only fitted statistics, finite transformed values and successful artifact reloads. It also records learned medians, modes, clipping bounds, encoding information, split summaries and full audit results.

[results/processed_dataset.csv](results/processed_dataset.csv) contains 48,805 records and 19 columns: 14 cleaned inputs, two derived features, income, split and source_row_id. It stores model inputs before scaling and one-hot encoding, with training then test records in source-row order. Serialization uses UTF-8, LF newlines, no DataFrame index and `%.12g` numeric formatting. The SHA256 of its exact bytes, also stored in `results/config_resolved.yaml`, is:

`9764686107f87c031ef8514c734c4786c037083625a5a89d24dc36c46633b91d`

The hash checks file identity, not data correctness. [data_card.md](data_card.md) records source, license, size, historical and demographic biases, and privacy concerns. Raw caches and local virtual environments are excluded from Git.

## 4. Three conclusions

1. On this fixed split, LogisticRegression achieved 0.8514 accuracy and 0.6517 F1 with the original predictors, compared with 0.8542 accuracy and 0.6610 F1 with the two derived features. The observed improvement is small and does not establish a statistically significant or universal gain.
2. Validation found 0 disallowed violations but identified 6,465 missing cells, 52 additional exact duplicate records and 5 conflicting feature profiles. These findings show why valid schemas alone cannot guarantee reliable labels; the manual sample-review status is reported explicitly above.
3. Group-aware splitting and training-only preprocessing preserve the held-out evaluation while making every transformation reproducible. Saved configuration, fitted artifacts, row assignments, checks and the processed-data hash allow the experiment to be rerun and inspected; the data card documents limits on generalization and use.
