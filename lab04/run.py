#!/usr/bin/env python3
"""Run every stage of Lab 4, from read-only ingestion to auditable artifacts."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import copy
import hashlib
import importlib.metadata
import platform
import warnings
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from src.preprocessing import CATEGORICAL, NUMERIC, make_preprocessor
from src.splitting import deduplicate_training, stratified_group_split
from src.validation import SCHEMA, label_audit, validate_frame

ROOT = Path(__file__).resolve().parent
LEGACY_ARTIFACTS = (
    'assisted_sample_review.md', 'duplicate_conflicts.csv', 'ingestion.json',
    'label_quality.json', 'label_quality.md', 'label_sample_50.csv',
    'leakage_checks.json', 'learned_statistics.json', 'metrics.json',
    'model_baseline.joblib', 'model_engineered.joblib', 'processed.sha256',
    'sample_checks.csv', 'split_summary.json', 'splits.csv', 'validation_report.json',
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(frame):
    """In-memory integrity check; independent of the exported CSV hash."""
    payload = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return hashlib.sha256(payload + str(frame.dtypes).encode()).hexdigest()


def remove_legacy_artifacts(results):
    """Remove only superseded generated outputs after new evidence is saved."""
    for name in LEGACY_ARTIFACTS:
        (results / name).unlink(missing_ok=True)


def validate_config(config):
    fixed = {
        ('dataset', 'name'): 'adult', ('dataset', 'version'): 2,
        ('dataset', 'positive_label'): '>50K',
        ('dataset', 'expected_rows'): 48842, ('dataset', 'expected_columns'): 15,
        ('split', 'strategy'): 'stratified_feature_groups',
        ('split', 'test_size'): 0.30,
        ('cleaning', 'numeric_imputation'): 'median',
        ('cleaning', 'categorical_imputation'): 'most_frequent',
        ('cleaning', 'missing_marker'): '?',
        ('cleaning', 'duplicate_policy'): 'exact_train_rows_only',
        ('cleaning', 'clip_lower'): 0.01, ('cleaning', 'clip_upper'): 0.99,
        ('model', 'decision_threshold'): 0.5,
        ('label_audit', 'sample_size'): 50,
    }
    for (section, key), value in fixed.items():
        if config[section][key] != value:
            raise ValueError(f'{section}.{key} must be {value!r} for this implementation.')


def write_validation_report(results, ingestion, validation, audit=None):
    """One readable report for ingestion, schema checks and the label audit."""
    summary = validation['summary']
    duplicate = validation['duplicates']
    lines = [
        '# Validation report', '',
        f"Raw ingestion: **{ingestion['rows']:,} rows × {ingestion['columns']} columns**. No rows dropped during validation.", '',
        'OpenML names the target `class`; validation uses an unmodified-value view named `income`. '
        'The cached source and original DataFrame retain the original name. `?` is allowed in source categorical vocabularies '
        'and interpreted as missing; OpenML has already parsed these tokens as nulls.', '',
        '| Check | Violations / findings |', '|---|---:|',
        f"| Schema/shape | {summary['structural_violations']} |",
        f"| Logical types | {summary['type_violation_cells']} |",
        f"| Numeric ranges | {summary['range_violation_cells']} |",
        f"| Unknown categories | {summary['category_violation_cells']} |",
        f"| Missing target labels | {summary['missing_target_rows']} |",
        f"| Allowed missing predictor cells | {summary['missing_predictor_cells']} |",
        f"| Exact duplicate additional records | {duplicate['exact_duplicate_additional_rows']} |",
        f"| Conflicting duplicate feature groups | {duplicate['conflicting_feature_groups']} |", '',
        '## Schema and ranges', '',
        '| Field | Logical type | Source dtype | Range / category count | Violations |',
        '|---|---|---|---|---:|',
    ]
    for column, kind in validation['schema']['logical_types'].items():
        bounds = validation['ranges'].get(column)
        allowed = validation['categories'].get(column, {}).get('allowed', [])
        policy = f"{bounds.get('min')}–{bounds.get('max') if bounds.get('max') is not None else 'unbounded'}" if bounds else f'{len(allowed)} allowed categories'
        violations = sum(validation[group].get(column, {}).get('violations', 0) for group in ['types', 'ranges', 'categories'])
        dtype = validation['schema']['storage_dtypes'].get(column, 'missing column')
        lines.append(f'| {column} | {kind} | {dtype} | {policy} | {violations} |')
    lines += ['', 'Integer fields are checked for finite integral values. Complete allowed-category '
              'vocabularies and machine-readable checks are stored in `config_resolved.yaml`.', '',
              '## Missing values', '',
        '| Column | Raw nulls | Literal `?` | Total missing |', '|---|---:|---:|---:|',
    ]
    for column, detail in validation['missing'].items():
        lines.append(f"| {column} | {detail['native_null_count']} | {detail['question_mark_count']} | {detail['total_missing']} |")
    lines += ['', 'Duplicates and allowed missing predictors are quality findings. '
              'A structural, type, range, unknown-category, or missing-target violation stops processing after the report is saved.']
    if validation['fatal']:
        lines += ['', '## Violations', '',
                  f"Missing columns: `{validation['schema']['missing_columns']}`. Unexpected columns: `{validation['schema']['unexpected_columns']}`.",
                  f"Expected row count matches: `{validation['shape']['row_count_matches']}`."]
        for group in ['types', 'ranges', 'categories', 'missing']:
            for column, check in validation[group].items():
                if check.get('violations', 0) or (group == 'missing' and not check['allowed'] and check['total_missing']):
                    lines.append(f"- {column} ({group}): `{check['examples']}`")
    if audit is not None:
        conflict = audit['duplicate_conflicts']
        sample = audit['sample']
        review = audit['manual_review']
        lines += ['', '## Label quality', '',
                  f"The full raw-data audit found **{conflict['groups']} conflicting feature groups / {conflict['rows']} rows**. "
                  'Each group has identical raw predictors and different recorded income labels. This does not establish which label is correct. No labels are reassigned.', '',
                  '| Conflict group | Source row ID | Income |', '|---|---:|---|']
        for row in conflict['records']:
            lines.append(f"| {row['conflict_group_id']} | {row['source_row_id']} | {row['income']} |")
        lines += ['', 'Full raw conflict records are saved under `label_audit.duplicate_conflicts.records` in `config_resolved.yaml`.', '',
                  f"A seeded sample of {sample['sampled_rows']} raw records (random_state={sample['seed']}) has "
                  f"{sample['impossible_value_count']} impossible-value or education-code inconsistencies detected automatically. "
                  'Checks cover dictionary categories, integer types, ranges, missing targets and education/education-num consistency. Allowed missing predictors are reported.', '',
                  f"**Human review: {review['status']} ({review['completed_rows']}/{sample['sampled_rows']} completed).** "
                  'All 50 source records, automated checks and reviewer/verdict/notes fields are combined in `manual_review.csv`. '
                  'Automated checks do not constitute a student manual review or verify true income.', '',
                  'Dictionary: [UCI Adult documentation](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names). '
                  "Adult uses recorded labels, so the alternative subjective-label relabeling and Cohen's kappa procedure does not apply."]
    (results / 'validation_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_reports(results, config, ingestion, validation, audit, split_info, leakage, metrics, manifest):
    summary = validation['summary']
    duplicate = validation['duplicates']
    missing = validation['missing']
    write_validation_report(results, ingestion, validation, audit)
    metric_rows = '\n'.join(f"| {r['variant']} | {r['accuracy']:.4f} | {r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} |" for r in metrics)
    delta_accuracy = metrics[1]['accuracy'] - metrics[0]['accuracy']
    delta_f1 = metrics[1]['f1'] - metrics[0]['f1']
    report = f"""# Lab 4: Data pipeline and quality

## 1. Goal

Build a reproducible data pipeline for mixed numerical and categorical inputs, validate data quality, prevent information leakage, and evaluate whether derived features improve an income classifier. The supervised binary-classification task predicts whether a person's recorded annual income exceeds US$50,000. The two outcome classes are `<=50K` and `>50K`, with `>50K` treated as the positive class. OpenML's target name `class` is represented as `income` in the lab's reports.

The 14 inputs are age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week and native-country. They combine six numerical and eight categorical variables. A row is an observation, not a guaranteed unique person. The census weighting field fnlwgt is used as an input rather than an evaluation weight. The exercise studies data engineering and predictive performance on historical records; it does not establish suitability for decisions about individuals.

Success is evaluated using accuracy and positive-class F1 on a fixed, stratified 30% test partition. Accuracy measures overall correctness, while F1 combines precision and recall for the less common higher-income class. LogisticRegression provides a baseline, compared with a predefined variant that adds net capital gain and weekly hours divided by age minus 16. Both variants use the same split and model settings. Their test results describe this experiment and are not used to tune the features, preprocessing or classifier.

The main constraints are reproducibility and prevention of information leakage. Splitting precedes cleaning, identical raw feature profiles stay together, and every test record is retained. Imputation, clipping, scaling and encoding learn only from training data. Validation must report missing values, duplicates, invalid values and label conflicts without silently deleting records. The saved transformer, configuration, processed-data SHA256, validation evidence and data card make the pipeline auditable and runnable with one command. Automated sample checks support, but do not replace, the student's manual inspection.

## 2. Method

The Adult dataset is downloaded using `fetch_openml(name='adult', version=2, as_frame=True)`, OpenML data ID {ingestion['openml_data_id']}. It contains {ingestion['rows']:,} records with 14 predictors (six numerical and eight categorical) and one income target, giving {ingestion['columns']} columns. OpenML names the target `class`; reports use `income`. Ingestion preserves the original values and cached source. The positive label is `>50K`.

Validation checks column names, logical types, category vocabularies, missing values, exact duplicates and conflicting labels. Required ranges are age 17–90, hours-per-week 1–99, capital-gain 0–99999 and capital-loss 0–4356. Additional checks require positive fnlwgt and education-num 1–16. The source token `?` means missing and is allowed in categorical vocabularies; OpenML represents these values as nulls. Validation reports findings without dropping rows; disallowed violations stop the pipeline.

A stratified 70/30 split with random_state={config['split']['random_state']} is made before cleaning. Identical raw 14-feature profiles are assigned together, including profiles with conflicting labels, while preserving exact stratified class-count targets. This prevents duplicate-profile overlap between training and test. Before deduplication, training class counts are `{split_info['train_class_counts']}` and test counts are `{split_info['test_class_counts']}`. Exact complete duplicates are then removed from training only; every test row is retained.

Numerical median and categorical mode imputers are fitted on deduplicated training data. All six original numerical features are then clipped to their training 1st/99th percentiles. StandardScaler and OneHotEncoder with `handle_unknown='ignore'` also fit on training only. Test features receive the fitted transformations; the original test rows, labels, order and count remain unchanged. Retained row assignments are included in the processed CSV; removed training source IDs and split summaries are saved in [results/config_resolved.yaml](results/config_resolved.yaml).

Two LogisticRegression pipelines use the same split and settings: solver={config['model']['solver']}, C={config['model']['C']}, max_iter={config['model']['max_iter']}, tol={config['model']['tol']}, random_state={config['model']['random_state']} and decision threshold={config['model']['decision_threshold']}. The baseline uses the original predictors. After imputation and clipping, the engineered variant adds `capital_net = capital-gain - capital-loss` and `hours_per_age = hours-per-week / (age - 16)`. The age constraint keeps the denominator positive. Neither model settings nor feature choices are tuned using test results.

Label quality is checked by finding identical raw feature profiles with different income labels and drawing {audit['sample']['sampled_rows']} random records with random_state={audit['sample']['seed']}. Sample checks cover dictionary categories, integer types, ranges and agreement between education and education-num. A separate worksheet supports the student's manual inspection.

Run the complete pipeline from the repository root with `./lab04/run.sh`. Settings are stored in [config.yaml](config.yaml), and the effective configuration, schema, environment versions and processed-data hash are saved in [results/config_resolved.yaml](results/config_resolved.yaml). The fitted engineered transformer is saved as [results/preprocessor.joblib](results/preprocessor.joblib). Model serialization is checked in memory without saving separate model files; the transformer is reloaded from disk. Reloaded transformations and predictions match their original equivalents.

## 3. Results

| Model variant | Test accuracy | F1 (>50K) | Precision (>50K) | Recall (>50K) |
|---|---:|---:|---:|---:|
{metric_rows}

Tabular metrics are saved in [results/metrics.csv](results/metrics.csv); full-precision metrics and confusion matrices are included in [results/config_resolved.yaml](results/config_resolved.yaml). Derived features change accuracy by {100 * delta_accuracy:+.3f} percentage points and F1 by {delta_f1:+.6f}. These are results from one predefined holdout; statistical significance has not been established. Retained test duplicates make this a record-level evaluation. No subgroup fairness claim is made.

### Data quality and partition sizes

| Check | Result |
|---|---:|
| Disallowed schema, type, range, category or required-label violations | {summary['total_violations']} |
| Missing predictor cells | {summary['missing_predictor_cells']:,} |
| Missing workclass / occupation / native-country cells | {missing['workclass']['total_missing']:,} / {missing['occupation']['total_missing']:,} / {missing['native-country']['total_missing']:,} |
| Additional exact duplicate records in the full raw dataset | {duplicate['exact_duplicate_additional_rows']} |
| Exact duplicate training rows removed | {leakage['train_duplicate_rows_removed']} |
| Raw feature groups shared by training and test | {leakage['shared_raw_feature_groups']} |
| Conflicting feature groups / affected rows | {audit['duplicate_conflicts']['groups']} / {audit['duplicate_conflicts']['rows']} |
| Sampled records / impossible values detected automatically | {audit['sample']['sampled_rows']} / {audit['sample']['impossible_value_count']} |

| Partition | Before training deduplication | Retained for modelling |
|---|---:|---:|
| Training | {split_info['train_rows_before_dedup']:,} | {leakage['train_rows_after_dedup']:,} |
| Test | {split_info['test_rows']:,} | {split_info['test_rows']:,} |

Detailed validation and label-quality findings are combined in [results/validation_report.md](results/validation_report.md). Full raw conflicting records are preserved in [results/config_resolved.yaml](results/config_resolved.yaml). Conflicting profiles establish recorded-label ambiguity; they do not identify which income label is correct. No labels are reassigned. **Human-review worksheet status: {audit['manual_review']['status']} ({audit['manual_review']['completed_rows']}/{audit['sample']['sampled_rows']} completed).** The 50 source records, automated row checks and human worksheet are combined in [results/manual_review.csv](results/manual_review.csv).

### Reproducibility evidence

[results/config_resolved.yaml](results/config_resolved.yaml) consolidates source integrity, preserved test rows, zero shared raw feature groups, training-only fitted statistics, finite transformed values and successful artifact reloads. It also records learned medians, modes, clipping bounds, encoding information, split summaries and full audit results.

[results/processed_dataset.csv](results/processed_dataset.csv) contains {manifest['processed_rows']:,} records and {manifest['processed_columns']} columns: 14 cleaned inputs, two derived features, income, split and source_row_id. It stores model inputs before scaling and one-hot encoding, with training then test records in source-row order. Serialization uses UTF-8, LF newlines, no DataFrame index and `%.12g` numeric formatting. The SHA256 of its exact bytes, also stored in `results/config_resolved.yaml`, is:

`{manifest['processed_sha256']}`

The hash checks file identity, not data correctness. [data_card.md](data_card.md) records source, license, size, historical and demographic biases, and privacy concerns. Raw caches and local virtual environments are excluded from Git.

## 4. Three conclusions

1. On this fixed split, LogisticRegression achieved {metrics[0]['accuracy']:.4f} accuracy and {metrics[0]['f1']:.4f} F1 with the original predictors, compared with {metrics[1]['accuracy']:.4f} accuracy and {metrics[1]['f1']:.4f} F1 with the two derived features. The observed improvement is small and does not establish a statistically significant or universal gain.
2. Validation found {summary['total_violations']} disallowed violations but identified {summary['missing_predictor_cells']:,} missing cells, {duplicate['exact_duplicate_additional_rows']} additional exact duplicate records and {audit['duplicate_conflicts']['groups']} conflicting feature profiles. These findings show why valid schemas alone cannot guarantee reliable labels; the manual sample-review status is reported explicitly above.
3. Group-aware splitting and training-only preprocessing preserve the held-out evaluation while making every transformation reproducible. Saved configuration, fitted artifacts, row assignments, checks and the processed-data hash allow the experiment to be rerun and inspected; the data card documents limits on generalization and use.
"""
    (ROOT / 'report.md').write_text(report, encoding='utf-8')

    card = f"""# Data card: Adult income, Lab 4

## Source, license and task

Source: [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult), also called Census Income, distributed through [OpenML Adult version 2, data ID 1590](https://www.openml.org/d/1590). UCI credits Barry Becker and Ronny Kohavi. Cite: Becker, B. and Kohavi, R. (1996), Adult, UCI Machine Learning Repository, [doi:10.24432/C5XW20](https://doi.org/10.24432/C5XW20). UCI currently publishes the dataset under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Attribute the source and indicate these imputation, clipping, deduplication and feature-engineering changes when redistributing.

This educational binary-classification exercise predicts recorded annual income above US$50,000 from historical US census attributes. The original data comes from a 1994 Census database. The source extraction selected AAGE >16, AGI >100, AFNLWGT >1 and HRSWK >0. These selection rules limit the covered population; AGI is not an available predictor and its extraction condition cannot be rechecked from these 14 features. See the [original dictionary](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names).

## Size and fields

Raw data: **{ingestion['rows']:,} records, 14 predictors and one target**. Numerical predictors: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week. Eight categorical predictors: workclass, education, marital-status, occupation, relationship, race, sex, native-country. Labels: `<=50K` and `>50K`, where `>50K` is positive. Raw class counts: `{ingestion['class_counts']}`. The OpenML target column is `class`; reports use `income`. `fnlwgt` is retained as an input, not a sample weight.

Processed data: **{manifest['processed_rows']:,} records** ({leakage['train_rows_after_dedup']:,} train; {split_info['test_rows']:,} test), 16 predictors including capital_net and hours_per_age, income label, split and source_row_id. Source identifiers are positional audit keys, not names or person identifiers. The CSV has {manifest['processed_columns']} columns and {manifest['processed_bytes']:,} bytes. Exact hash, serialization details and complete audit evidence are in `results/config_resolved.yaml`.

## Quality and preparation

Validation found {summary['total_violations']} schema/type/range/category/required-label violations. Missing predictor cells: {summary['missing_predictor_cells']} (workclass {missing['workclass']['total_missing']}, occupation {missing['occupation']['total_missing']}, native-country {missing['native-country']['total_missing']}). `?` denotes missing; OpenML represents it as a native null. Training medians/modes impute missing predictors. Each original numeric feature is clipped to training 1st/99th percentiles. This can suppress legitimate extremes, especially rare capital events. Numeric scaling and category encoding use training statistics only. Derived features are computed from these cleaned values.

Feature-group-aware stratification keeps identical raw profiles in one partition while preserving exact 70/30 class-count targets. It produces zero raw-profile overlap. {leakage['train_duplicate_rows_removed']} exact complete training duplicates are removed. Test duplicates are retained as required. The full data contains {audit['duplicate_conflicts']['groups']} conflicting feature groups ({audit['duplicate_conflicts']['rows']} rows); no claim is made about which label is correct. Labels are not altered. Automated checks of the 50-row sample found {audit['sample']['impossible_value_count']} impossible values. The human worksheet in `results/manual_review.csv` reports status `{audit['manual_review']['status']}` ({audit['manual_review']['completed_rows']}/{audit['sample']['sampled_rows']} completed).

## Known biases and limits

Historical income reflects social and economic disparities. Representation is uneven across demographic and income groups, and historical race/sex category vocabularies are limited. Occupation, education, family relationships and country can act as demographic proxies. Mode imputation can overrepresent majority categories, and missingness may be systematic. The >50K threshold is historical, without inflation adjustment. These observations do not establish current population behavior, fairness, causality, or fitness for high-stakes decisions. Aggregate accuracy/F1 do not substitute for subgroup evaluation. The single held-out split and retained duplicate test records limit statistical independence and generalization. Source extraction and the use of fnlwgt as a predictor further limit population-level conclusions.

## Privacy and appropriate use

No direct names or addresses are included, but combinations of age, country, work and family attributes can act as quasi-identifiers. Race, sex, income and other attributes warrant care. Do not attempt reidentification or link records to identified people. Use these artifacts for education and reproducibility; deployment in employment, lending, insurance or other consequential decisions would require a separate justification, privacy assessment and fairness analysis. This lab does not authorize such deployment.

## Reproduction and maintenance

Run `./lab04/run.sh` from the repository root. All random seeds, model settings, clipping quantiles, schema, source version, environment versions, row assignments and hashes are saved. The first run requires internet; the cached source supports later offline Python runs. Artifacts depend on the recorded package versions and have not been validated across arbitrary dependency or platform changes. Human review status must be updated honestly; a script cannot attest to a student's inspection.
"""
    (ROOT / 'data_card.md').write_text(card, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    validate_config(config)
    results = ROOT / config['paths']['results']
    results.mkdir(parents=True, exist_ok=True)
    cache = ROOT / config['paths']['cache']
    print('Ingestion: fetching/caching Adult v2 without modifying raw values...', flush=True)
    dataset = fetch_openml(name=config['dataset']['name'], version=config['dataset']['version'], as_frame=True, data_home=str(cache))
    raw = dataset.frame
    original_fingerprint = fingerprint(raw)
    ingestion = {
        'name': 'adult', 'version': 2, 'openml_data_id': str(dataset.details['id']),
        'rows': len(raw), 'columns': len(raw.columns), 'predictors': len(dataset.feature_names),
        'source_target_name': dataset.target.name, 'logical_target_name': 'income',
        'storage_dtypes': {k: str(v) for k, v in raw.dtypes.items()},
        'class_counts': {str(k): int(v) for k, v in dataset.target.value_counts().items()},
        'raw_frame_fingerprint': original_fingerprint, 'raw_rows_dropped': 0,
    }
    print(f"Ingestion: {ingestion['rows']} rows, {ingestion['columns']} columns", flush=True)
    # This view changes only the target name. The fetched frame remains untouched.
    frame = raw.rename(columns={dataset.target.name: 'income'}).copy(deep=True)
    schema = copy.deepcopy(SCHEMA)
    schema['expected_rows'] = config['dataset']['expected_rows']
    validation = validate_frame(frame, schema)
    if ingestion['columns'] != config['dataset']['expected_columns'] or validation['fatal']:
        write_validation_report(results, ingestion, validation)
        failed = {'status': 'validation_failed', 'config': config, 'schema': schema, 'ingestion': ingestion, 'validation': validation}
        (results / 'config_resolved.yaml').write_text(yaml.safe_dump(failed, sort_keys=False), encoding='utf-8')
        raise ValueError('Validation failed: inspect results/validation_report.md; no rows silently dropped.')
    audit_config = config['label_audit']
    audit = label_audit(frame, schema, audit_config['random_state'], audit_config['sample_size'], results)

    train_raw, test_raw, split_info = stratified_group_split(
        frame, target='income', test_size=config['split']['test_size'], seed=config['split']['random_state'])
    test_fingerprint = fingerprint(test_raw)
    train, dropped = deduplicate_training(train_raw)
    split_info['removed_training_source_row_ids'] = [int(index) for index in train_raw.index if index not in train.index]
    X_train = train.drop(columns='income')
    X_test = test_raw.drop(columns='income')
    y_train = train['income'].eq(config['dataset']['positive_label']).astype(int)
    y_test = test_raw['income'].eq(config['dataset']['positive_label']).astype(int)
    metrics, fitted = [], {}
    print(f'Split: {len(train_raw)} train → {len(train)} after train-only dedup; {len(test_raw)} test unchanged.', flush=True)
    for engineered, variant in [(False, 'baseline'), (True, 'engineered')]:
        preprocessor = make_preprocessor(engineered, config['cleaning']['clip_lower'], config['cleaning']['clip_upper'])
        model_settings = {k: v for k, v in config['model'].items() if k != 'decision_threshold'}
        pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(**model_settings))])
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        metric = {
            'variant': variant, 'accuracy': float(accuracy_score(y_test, prediction)),
            'f1': float(f1_score(y_test, prediction, zero_division=0)),
            'precision': float(precision_score(y_test, prediction, zero_division=0)),
            'recall': float(recall_score(y_test, prediction, zero_division=0)),
            'confusion_matrix_labels_0_1': confusion_matrix(y_test, prediction, labels=[0, 1]).tolist(),
            'train_rows': len(train), 'test_rows': len(test_raw),
            'encoded_features': int(preprocessor.transform(X_train.iloc[:1]).shape[1]),
            'n_iter': int(pipeline['classifier'].n_iter_[0]),
        }
        serialized_model = BytesIO()
        joblib.dump(pipeline, serialized_model)
        serialized_model.seek(0)
        restored = joblib.load(serialized_model)
        np.testing.assert_array_equal(restored.predict(X_test), prediction)
        fitted[variant] = pipeline
        metrics.append(metric)
        print(f"{variant}: accuracy={metric['accuracy']:.6f}, F1={metric['f1']:.6f}", flush=True)
    pd.DataFrame([{k: v for k, v in row.items() if k != 'confusion_matrix_labels_0_1'} for row in metrics]).to_csv(results / 'metrics.csv', index=False, float_format='%.12g')

    preprocessor = fitted['engineered']['preprocessor']
    joblib.dump(preprocessor, results / 'preprocessor.joblib')
    reloaded = joblib.load(results / 'preprocessor.joblib')
    expected = preprocessor.transform(X_test)
    actual = reloaded.transform(X_test)
    assert (actual != expected).nnz == 0
    assert np.isfinite(expected.data).all()
    assert np.isfinite(preprocessor.transform(X_train).data).all()
    imputer = preprocessor['impute']
    clipper = preprocessor['clip']
    statistics = {
        'fitted_on': 'deduplicated training partition only', 'training_rows': len(train),
        'numeric_medians': dict(zip(NUMERIC, imputer.numeric_.statistics_.tolist())),
        'categorical_modes': dict(zip(CATEGORICAL, imputer.categorical_.statistics_.tolist())),
        'clip_lower': clipper.lower_bounds_.to_dict(), 'clip_upper': clipper.upper_bounds_.to_dict(),
        'standard_scaler_means': preprocessor['encode'].named_transformers_['numeric'].mean_.tolist(),
        'standard_scaler_samples_seen': int(preprocessor['encode'].named_transformers_['numeric'].n_samples_seen_),
        'encoded_feature_names': preprocessor['encode'].get_feature_names_out().tolist(),
        'one_hot_categories': {name: categories.tolist() for name, categories in zip(CATEGORICAL, preprocessor['encode'].named_transformers_['categorical'].categories_)},
    }

    cleaned_train = preprocessor[:-1].transform(X_train)
    cleaned_test = preprocessor[:-1].transform(X_test)
    cleaned_train.insert(0, 'source_row_id', cleaned_train.index)
    cleaned_test.insert(0, 'source_row_id', cleaned_test.index)
    cleaned_train.insert(0, 'split', 'train')
    cleaned_test.insert(0, 'split', 'test')
    cleaned_train['income'] = train['income'].astype(object)
    cleaned_test['income'] = test_raw['income'].astype(object)
    processed = pd.concat([cleaned_train, cleaned_test], ignore_index=True)
    processed_path = results / 'processed_dataset.csv'
    processed.to_csv(processed_path, index=False, encoding='utf-8', lineterminator='\n', float_format='%.12g')
    processed_hash = sha256(processed_path)

    leakage = {
        'raw_frame_unchanged': fingerprint(raw) == original_fingerprint,
        'raw_test_unchanged': fingerprint(test_raw) == test_fingerprint,
        'all_test_rows_retained': len(cleaned_test) == len(test_raw),
        'train_test_source_ids_disjoint': not bool(set(train_raw.index) & set(test_raw.index)),
        'all_raw_rows_assigned': len(train_raw) + len(test_raw) == len(frame),
        'shared_raw_feature_groups': split_info['feature_groups_crossing_split'],
        'preprocessor_fit_rows': statistics['standard_scaler_samples_seen'],
        'train_rows_after_dedup': len(train), 'train_duplicate_rows_removed': dropped,
        'train_numeric_medians_match': bool(np.allclose(X_train[NUMERIC].median().to_numpy(), imputer.numeric_.statistics_)),
        'train_categorical_modes_match': all(X_train[c].astype(object).replace('?', np.nan).dropna().mode().iloc[0] == statistics['categorical_modes'][c] for c in CATEGORICAL),
        'preprocessor_reload_matches': True, 'model_reload_predictions_match': True,
        'transformed_values_finite': True,
    }
    assert all(leakage[key] for key in ['raw_frame_unchanged', 'raw_test_unchanged', 'all_test_rows_retained', 'train_test_source_ids_disjoint', 'all_raw_rows_assigned', 'train_numeric_medians_match', 'train_categorical_modes_match'])
    assert leakage['shared_raw_feature_groups'] == 0
    assert statistics['standard_scaler_samples_seen'] == len(train)
    environment = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib', 'PyYAML', 'threadpoolctl']},
        'numerical_threads': 1,
    }
    manifest = {
        'config': config, 'schema': schema, 'ingestion': ingestion, 'environment': environment,
        'processed_file': 'processed_dataset.csv', 'processed_sha256': processed_hash,
        'processed_rows': len(processed), 'processed_columns': len(processed.columns),
        'processed_bytes': processed_path.stat().st_size,
        'serialization': {'encoding': 'UTF-8', 'newline': 'LF', 'float_format': '%.12g', 'index': False, 'ordering': 'train then test; source_row_id ascending within each split', 'stage': 'imputed/clipped/engineered before scaling and encoding'},
        'feature_formulas': {'capital_net': 'capital-gain - capital-loss', 'hours_per_age': 'hours-per-week / (age - 16)'},
        'validation': validation, 'split_summary': split_info, 'leakage_checks': leakage,
        'learned_statistics': statistics, 'metrics': metrics, 'label_audit': audit,
    }
    (results / 'config_resolved.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding='utf-8')
    write_reports(results, config, ingestion, validation, audit, split_info, leakage, metrics, manifest)
    remove_legacy_artifacts(results)
    print(f'SHA256: {processed_hash}\nEvidence written to {results}', flush=True)


if __name__ == '__main__':
    main()
