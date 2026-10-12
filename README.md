# ML Systems laboratory repository

Lab 1 only: implementation and measured results.

## Run Lab 1

Use **Python 3.11.8**. From this repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python lab01/src/print_versions.py
python lab01/src/benchmark.py
```

### Dependency conflict in the supplied manual

The exact supplied pins are inconsistent: `mlflow==2.14.1` declares `pyarrow>=4.0.0,<16`, but the manual requires `pyarrow==16.1.0`. A normal installation of `requirements.txt` fails. The environment used for these measurements preserves all manual pins using this explicit workaround:

```bash
python -c "from pathlib import Path; Path('requirements-resolve.tmp').write_text(Path('requirements.txt').read_text().replace('pyarrow==16.1.0', 'pyarrow==15.0.2'))"
python -m pip install -r requirements-resolve.tmp
python -m pip install --no-deps pyarrow==16.1.0
rm requirements-resolve.tmp
```

This does **not** resolve the declared incompatibility. `pip check` reports it; MLflow/PyArrow interoperability was not validated. Neither package is used in Lab 1. Ask the instructor which pin to change before using those packages in later labs. `requirements-installed.txt` records the complete environment used.

## Layout

- `lab01/src/`: version verification and reproducible benchmark scripts.
- `lab01/results/`: accuracy, timing, memory, raw samples, host information, and two serialized models (one subfolder each to avoid overwriting `model.joblib`).
- `lab01/report.md`: report based on actual measurements.
- `data/`: ignored by Git; the Lab 1 dataset is bundled with scikit-learn.

## Measurement scope

Benchmarks run on the local CPU with numerical-library threads set to one. They measure this Python/scikit-learn process, not target-device performance. Memory includes the Python runtime and imported libraries, including PyTorch required by the seed instructions. RSS is sampled approximately every 1 ms; very short peaks may be missed. Separate fresh processes measure training and inference RSS so sampling overhead does not contaminate timing results.


Assigned reading (not performed by these scripts): Reddi, Chapter 1 §§1.4–1.6; Chapter 2 §§2.2–2.5, §§2.8–2.9; Chapter 12 §§12.4, 12.7.
