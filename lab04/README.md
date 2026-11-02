# Lab 4: Data pipeline and quality

This lab builds a reproducible Adult income pipeline, validates its data, compares logistic regression with and without derived features, and saves the evidence required by the assignment. The original Lab 1 files remain in `lab01/`.

## Run

From the repository root:

```bash
./lab04/run.sh
```

The script creates an isolated virtual environment, installs `lab04/requirements.txt`, and runs the pipeline. Python 3.11 is recommended. The first run needs internet access to download the Adult dataset from OpenML and install dependencies. Results are written to `lab04/results/`.

With the dependencies already installed, the Python entry point can also be run directly:

```bash
python lab04/run.py
python lab04/run.py --config lab04/config.yaml
```

The fixed random seed is 42. Keep the configuration, dependency versions, and fetched dataset version together when reproducing the recorded experiment.

Optional focused test suite, using the environment created by `run.sh`:

```bash
lab04/.venv/bin/python -m pip install -r lab04/requirements-dev.txt
lab04/.venv/bin/python -m pytest lab04/tests -q
```

To load the serialized transformer in a separate Python process, make its source package importable:

```python
import sys
from pathlib import Path
import joblib

sys.path.insert(0, str(Path("lab04").resolve()))
preprocessor = joblib.load("lab04/results/preprocessor.joblib")
# preprocessor.transform(raw_feature_frame) expects the 14 named input columns.
```

## What the pipeline does

1. Fetches `fetch_openml(name='adult', version=2, as_frame=True)` and records the untouched ingestion shape: 48,842 rows and 15 columns. OpenML's target name `class` is interpreted as the report field `income`. Numerical fields are checked for integer-valued content even when OpenML exposes a floating-point dtype.
2. Checks column names, logical types, allowed categories, ranges, missing values, duplicates, and conflicting income labels. The literal `?` is an allowed source token that means missing; existing nulls are also treated as missing. Validation reports violations instead of silently dropping rows.
3. Makes a reproducible, stratified 70/30 training/test split before cleaning: 34,189 training rows and 14,653 test rows. Identical raw 14-feature profiles, including profiles with conflicting labels, are assigned together. Seeded group assignment and singleton allocation preserve the exact stratified class-count targets while preventing feature-profile overlap between partitions. Exact duplicate complete records are then removed from training only. All original test rows remain available for evaluation.
4. Fits categorical mode and numerical median imputers on training data, then learns numerical 1st/99th percentile clipping thresholds from training data. Scaling and one-hot encoding also fit on training only.
5. Compares two logistic regression pipelines using the same split and model settings. The second adds `capital_net = capital-gain - capital-loss` and `hours_per_age = hours-per-week / (age - 16)` after imputation and clipping. Accuracy and positive-class F1 are recorded for both variants. No parameter choice is made using the test results.
6. Audits feature profiles shared by training and test and verifies that their intersection is empty. The test partition is never deduplicated or used to learn preprocessing statistics.
7. Exports the fitted preprocessing transformer, processed data, metrics, validation evidence, and a 50-row review worksheet. The effective YAML configuration also stores the processed-data SHA256 and detailed audit evidence. The report and data card are generated in `lab04/`.

The Goal section of [report.md](report.md) contains the problem definition, inputs, success criteria, and constraints.

## Evidence and review

The generated [report.md](report.md) records the experiment and metrics; [data_card.md](data_card.md) describes the source, license, size, known biases, and privacy concerns. A normal run writes six files in `results/`:

| File | Evidence |
| --- | --- |
| `metrics.csv` | Accuracy, F1, precision, and recall for the baseline and engineered models |
| `validation_report.md` | Original shape, full schema and ranges, missing-value and duplicate findings, and label-quality audit with conflicting records |
| `preprocessor.joblib` | Fitted engineered preprocessing transformer requested by the assignment |
| `processed_dataset.csv` | Cleaned train/test records with source row IDs, split assignments, derived features, and income labels; before scaling and one-hot encoding |
| `config_resolved.yaml` | Effective settings, schema, source metadata, dependency versions, processed-data SHA256, full metrics, split summary, leakage checks, learned statistics, and label audit including raw conflict records |
| `manual_review.csv` | The 50 sampled raw records with source row IDs, automated checks, and human reviewer fields |

Preserve the processed CSV together with `config_resolved.yaml` when submitting the lab. The SHA256 in that YAML identifies the exact CSV bytes. From the repository root, verify it using the environment created by `run.sh`:

```bash
lab04/.venv/bin/python - <<'PY'
import hashlib
from pathlib import Path
import yaml

results = Path("lab04/results")
config = yaml.safe_load((results / "config_resolved.yaml").read_text())
actual = hashlib.sha256((results / "processed_dataset.csv").read_bytes()).hexdigest()
assert actual == config["processed_sha256"], "Processed CSV hash mismatch"
print("Processed CSV SHA256 verified:", actual)
PY
```

The reproducible sample, automated row checks, and human worksheet are combined in `results/manual_review.csv`. Its `reviewer`, `verdict`, and `notes` fields are blank until a person completes the review; the initial status is `pending_user_review`. Automated dictionary checks support that review, but they do not constitute a student's manual inspection. Open all 50 rows, compare their values with the [UCI Adult documentation](https://archive.ics.uci.edu/dataset/2/adult), and complete the reviewer fields before claiming that the manual part of requirement 6 is finished. The conflict audit is included in `results/validation_report.md` and `results/config_resolved.yaml`. A duplicate-label conflict identifies contradictory records; it does not determine which income label is correct. Missing target labels stop the pipeline because they cannot be imputed responsibly for supervised evaluation.

The dataset describes historical US census records and includes sensitive demographic attributes. Test performance on these records does not demonstrate fairness, causal relationships, or suitability for current decisions. The generated data card discusses these limitations in more detail.
