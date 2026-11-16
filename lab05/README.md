# Lab 5: Automatic differentiation and framework execution

This lab implements a scalar reverse-mode differentiation engine, checks its gradients, trains the same small classifier with that engine and PyTorch, and measures graph execution and tensor-layout costs. [report.md](report.md) presents the Goal, Method, Results, and Three conclusions in the same format as the previous labs.

## Run

From the repository root:

```bash
./lab05/run.sh
```

The script creates `lab05/.venv`, installs the pinned dependencies in `lab05/requirements.txt`, and runs the experiment. Python 3.11 is required for the recorded environment. The first installation needs internet access. The synthetic dataset is generated locally.

With the environment already installed:

```bash
lab05/.venv/bin/python lab05/run.py
lab05/.venv/bin/python lab05/run.py --config lab05/config.yaml
```

Optional focused tests:

```bash
lab05/.venv/bin/python -m pip install -r lab05/requirements-dev.txt
lab05/.venv/bin/python -m pytest lab05/tests -q
```

## Experiment

`src/value.py` implements scalar arithmetic, activations, and reverse-mode backpropagation using the chain rule. Iterative topological traversal supports shared intermediate nodes and repeated operands. Each `backward()` call resets gradients throughout its reachable graph before computing a fresh derivative.

The training dataset is `make_moons(n_samples=200, noise=0.20, random_state=42)`. A stratified 70/30 split gives 140 training and 60 test records. StandardScaler fits on training data only. Both implementations use the same 33-parameter network, `2 → 8 ReLU → 1 sigmoid`, shared Xavier-initialized weights, mean binary cross-entropy, and full-batch SGD for 200 epochs with learning rate 0.15. No settings are selected using test results.

Training uses CPU float64 with numerical thread limits set to one. A separate 20-epoch warm-up model is discarded for each implementation; five fresh measured models then start from the same weights. Their execution order alternates. Training time includes gradient reset, forward computation, loss, backward computation, and parameter updates. Initialization, data conversion, warm-up, and test evaluation are excluded. Parameter values, loss histories, and test probabilities are checked for numerical agreement.

The execution experiment compares eager inference with a model prepared using `torch.jit.trace` followed by `torch.jit.freeze`. Preparation time is recorded separately. For each batch size, 1 and 64, inference uses `no_grad`, 100 warm-up calls, and five timed loops of 1,000 calls. Results are milliseconds per call, averaged within each loop and summarized by the median across loops. Tracing records the operations exercised by example inputs; its control-flow limitations are described in the [PyTorch 2.3 trace documentation](https://docs.pytorch.org/docs/2.3/generated/torch.jit.trace.html). See also the [TorchScript reference](https://docs.pytorch.org/docs/2.3/jit.html).

The layout experiment reduces identical 512×512 values with `sum(dim=1)`: a non-contiguous `storage[:, ::2]` view with strides `(1024, 2)`, its prebuilt contiguous copy with strides `(512, 1)`, and a copy-then-sum operation. The third measurement includes copying on every call. Each variant uses 30 warm-up calls and five timed loops of 200 calls. The [PyTorch 2.3 tensor-view documentation](https://docs.pytorch.org/docs/2.3/tensor_view.html) explains shared storage and when `contiguous()` copies data.

## Saved evidence

A normal run writes five files in `lab05/results/`:

| File | Contents |
| --- | --- |
| `gradient_check.csv` | Gradient comparisons and numerical errors |
| `training.csv` | Median training times, losses, accuracy, and F1 for both implementations |
| `framework_benchmarks.csv` | Eager/traced inference and tensor-layout timing summaries |
| `graphs.txt` | The scalar expression graph, eager autograd graph, and frozen traced graph |
| `config_resolved.yaml` | Effective settings, versions, machine details, raw repeated timings, and correctness checks |

## Interpreting the measurements

The training comparison includes Python scalar graph construction and traversal versus vectorized PyTorch tensor operations. Its speed ratio measures these complete implementations, so it cannot isolate the cost of automatic differentiation alone. Repeated models share the same data and initialization; they measure timing variation, rather than uncertainty across datasets or random initializations.

A traced graph or a contiguous copy is not guaranteed to improve performance. Compare steady-state inference with its preparation cost, and prebuilt-copy reductions with copy-then-sum before deciding whether conversion pays off. These are host-specific measurements of a small CPU workload; the recorded versions and thread settings matter. PyTorch documents the role of those settings in [CPU threading and TorchScript inference](https://docs.pytorch.org/docs/2.3/notes/cpu_threading_torchscript_inference.html).
