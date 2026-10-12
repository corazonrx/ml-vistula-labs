# Lab 1: Environment and first system measurements


## 1. Goal

Build a reproducible Python environment, train linear and ensemble baselines, measure accuracy and system cost, and assess deployment feasibility against the supplied resource budgets.

## 2. Method

Python 3.11.8 and all 25 direct dependency versions match the manual. The manual's MLflow/PyArrow dependency conflict required an explicit installation override; the environment therefore has a known dependency inconsistency. Details are in README.md and results/dependency_check.txt. The THOP metadata version uses an equivalent post-release spelling. Version verification is saved in results/versions.txt.

The built-in Breast Cancer Wisconsin (Diagnostic) dataset contains 569 examples with 30 features. A stratified 70/30 split with random_state=42 gives 398 training and 171 test examples. No external dataset download is needed.

LogisticRegression uses max_iter=1000 and random_state=42 inside a StandardScaler pipeline. The scaler is fitted only on training data; its fitting, transformation, and storage are included in the measured system costs. RandomForestClassifier uses n_estimators=100 and random_state=42, with the other parameters at their defaults. Python, NumPy, and PyTorch seeds are set to 42, including the requested CUDA seed call. Computation uses the CPU, with numerical-library thread limits set to one.

Training is warmed up once using a separate model, then five fresh models are fitted and the median fit time is reported. The final fitted model is evaluated on the held-out test set. Single-sample inference is warmed up once, then predict is timed 100 times using the same held-out feature row, without its label; the median is reported. Timing uses perf_counter_ns. Accuracy is not used for tuning.

Models are serialized without compression into separate model subfolders, each containing model.joblib, so neither overwrites the other. Reloaded predictions were checked against the original models. File sizes include the scaler for logistic regression.

Memory is measured in fresh subprocesses, separately for training and inference, by sampling absolute process RSS approximately every 1 ms, including readings at operation boundaries. Inference memory is sampled over 100 predictions after one warm-up. It includes Python and imported libraries, including PyTorch, rather than only model allocations. These are sampled peaks; short-lived peaks may be missed. Memory sampling is separate from latency timing.

The host is macOS on Apple arm64; further machine details are saved in results/machine.json. KB and MB below denote 1024 bytes and 1024² bytes respectively. Binary versus decimal units do not change any pass/fail decision here.

## 3. Results

| Model | Test accuracy | Train median (ms) | Single-sample median (ms) | Size (bytes) | Size (KB) | Train peak RSS (MB) | Inference peak RSS (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.9883 | 1.0876 | 0.0346 | 2302 | 2.248 | 285.375 | 285.875 |
| random_forest | 0.9357 | 57.1524 | 0.8271 | 290905 | 284.087 | 285.469 | 285.578 |

Raw individual timings and memory samples' peak summaries are preserved in results/raw_measurements.json. Accuracy to four decimal places is also in results/baseline_accuracy.csv.

### Resource budgets from the manual

| Platform | Available memory | Maximum latency | Maximum model size |
|---|---:|---:|---:|
| Cloud | ≥ 1 GB | 100 ms | 500 MB |
| Edge | 256–1024 MB | 50 ms | 50 MB |
| Mobile | 64–256 MB | 20 ms | 10 MB |
| TinyML | ≤ 256 KB | 10 ms | 100 KB |

Available memory describes the device budget, not a minimum amount a model must consume. For a range, a model may fit some devices but not all.

### Deployment fit

| Platform | Logistic regression | Random forest |
|---|---|---|
| Cloud | Fits the measured resource limits | Fits the measured resource limits |
| Edge | Fits devices with sufficient memory above the measured ~286 MB process RSS; does not fit 256 MB | Fits devices with sufficient memory above the measured ~286 MB process RSS; does not fit 256 MB |
| Mobile | Current Python process exceeds the 256 MB maximum | Current Python process exceeds the 256 MB maximum |
| TinyML | Current runtime exceeds 256 KB RAM, despite the small serialized model | Exceeds both the 100 KB model-size limit and the 256 KB RAM budget |

Both models' measured single-sample latencies are below all four latency thresholds on this host. Both serialized models fit Cloud, Edge, and Mobile size budgets. Only the logistic-regression artifact is below the TinyML size limit. Memory is therefore the limiting constraint for Mobile and, for logistic regression, TinyML in the measured implementation. A practical deployment also needs memory headroom for the operating system and other services.

These decisions are preliminary resource comparisons, not demonstrations on target hardware. A portable or embedded implementation could use a different runtime and consume much less memory, particularly without PyTorch. Such a change requires new measurements; the current results cannot establish target-device latency or feasibility.

## 4. Three conclusions

1. On this fixed split, logistic regression with training-only standardization achieved 0.9883 accuracy, compared with 0.9357 for the 100-tree random forest. This is evidence for this dataset and split, not a universal ranking of the algorithms.
2. Logistic regression was faster to train and to make a single prediction, and its serialized pipeline was much smaller: about 2.25 KB versus 284.09 KB. It is the stronger baseline among these two measured configurations.
3. Small model files do not guarantee small runtime memory usage. Both Python processes used roughly 286 MB, making Cloud and sufficiently provisioned Edge devices plausible, while the current implementation exceeded Mobile and TinyML RAM budgets. Target-hardware validation is still necessary.
