# Data card: Adult income, Lab 4

## Source, license and task

Source: [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult), also called Census Income, distributed through [OpenML Adult version 2, data ID 1590](https://www.openml.org/d/1590). UCI credits Barry Becker and Ronny Kohavi. Cite: Becker, B. and Kohavi, R. (1996), Adult, UCI Machine Learning Repository, [doi:10.24432/C5XW20](https://doi.org/10.24432/C5XW20). UCI currently publishes the dataset under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Attribute the source and indicate these imputation, clipping, deduplication and feature-engineering changes when redistributing.

This educational binary-classification exercise predicts recorded annual income above US$50,000 from historical US census attributes. The original data comes from a 1994 Census database. The source extraction selected AAGE >16, AGI >100, AFNLWGT >1 and HRSWK >0. These selection rules limit the covered population; AGI is not an available predictor and its extraction condition cannot be rechecked from these 14 features. See the [original dictionary](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names).

## Size and fields

Raw data: **48,842 records, 14 predictors and one target**. Numerical predictors: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week. Eight categorical predictors: workclass, education, marital-status, occupation, relationship, race, sex, native-country. Labels: `<=50K` and `>50K`, where `>50K` is positive. Raw class counts: `{'<=50K': 37155, '>50K': 11687}`. The OpenML target column is `class`; reports use `income`. `fnlwgt` is retained as an input, not a sample weight.

Processed data: **48,805 records** (34,152 train; 14,653 test), 16 predictors including capital_net and hours_per_age, income label, split and source_row_id. Source identifiers are positional audit keys, not names or person identifiers. The CSV has 19 columns and 6,561,326 bytes. Exact hash, serialization details and complete audit evidence are in `results/config_resolved.yaml`.

## Quality and preparation

Validation found 0 schema/type/range/category/required-label violations. Missing predictor cells: 6465 (workclass 2799, occupation 2809, native-country 857). `?` denotes missing; OpenML represents it as a native null. Training medians/modes impute missing predictors. Each original numeric feature is clipped to training 1st/99th percentiles. This can suppress legitimate extremes, especially rare capital events. Numeric scaling and category encoding use training statistics only. Derived features are computed from these cleaned values.

Feature-group-aware stratification keeps identical raw profiles in one partition while preserving exact 70/30 class-count targets. It produces zero raw-profile overlap. 37 exact complete training duplicates are removed. Test duplicates are retained as required. The full data contains 5 conflicting feature groups (10 rows); no claim is made about which label is correct. Labels are not altered. Automated checks of the 50-row sample found 0 impossible values. The human worksheet in `results/manual_review.csv` reports status `pending_user_review` (0/50 completed).

## Known biases and limits

Historical income reflects social and economic disparities. Representation is uneven across demographic and income groups, and historical race/sex category vocabularies are limited. Occupation, education, family relationships and country can act as demographic proxies. Mode imputation can overrepresent majority categories, and missingness may be systematic. The >50K threshold is historical, without inflation adjustment. These observations do not establish current population behavior, fairness, causality, or fitness for high-stakes decisions. Aggregate accuracy/F1 do not substitute for subgroup evaluation. The single held-out split and retained duplicate test records limit statistical independence and generalization. Source extraction and the use of fnlwgt as a predictor further limit population-level conclusions.

## Privacy and appropriate use

No direct names or addresses are included, but combinations of age, country, work and family attributes can act as quasi-identifiers. Race, sex, income and other attributes warrant care. Do not attempt reidentification or link records to identified people. Use these artifacts for education and reproducibility; deployment in employment, lending, insurance or other consequential decisions would require a separate justification, privacy assessment and fairness analysis. This lab does not authorize such deployment.

## Reproduction and maintenance

Run `./lab04/run.sh` from the repository root. All random seeds, model settings, clipping quantiles, schema, source version, environment versions, row assignments and hashes are saved. The first run requires internet; the cached source supports later offline Python runs. Artifacts depend on the recorded package versions and have not been validated across arbitrary dependency or platform changes. Human review status must be updated honestly; a script cannot attest to a student's inspection.
