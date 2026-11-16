# Lab 5: Frameworks and automatic differentiation

## 1. Goal

Implement a scalar automatic-differentiation engine that constructs computation graphs and performs reverse-mode differentiation with correct gradient accumulation. Validate derivatives against PyTorch and numerical estimates, train the same small classifier in both engines, and measure eager versus traced execution and the effect of tensor contiguity.

## 2. Method

The custom `Value` node stores a scalar value, gradient, ordered parent references, operation name and local derivative rule. Addition, multiplication, constant powers and ReLU construct graph nodes during the forward pass. Subtraction, division, exponentials, logarithms and sigmoid support the classifier and binary cross-entropy (BCE). The backward pass obtains an iterative topological ordering, resets reachable gradients, seeds the scalar output gradient with 1, and traverses in reverse while adding each chain-rule contribution to parent gradients. Repeated operands and shared intermediate nodes accumulate their contributions. ReLU uses derivative zero at zero. A repeated `backward()` starts a fresh gradient calculation; this differs from PyTorch leaf-gradient accumulation across separate calls. Local derivative closures avoid output-reference cycles.

The required expression `((a*b + c)**2).relu().sum()` is evaluated using 5 elements per input vector. Python scalar nodes are summed to obtain the same scalar objective as PyTorch float64 tensors. Every a, b and c input derivative is compared with PyTorch autograd and central finite differences using epsilon=1e-06. Each maximum absolute error must be strictly below 1e-06. Additional tests cover shared branches, repeated operands, powers, ReLU, domain errors and deep graphs.

The generated `make_moons` dataset contains 200 samples with noise=0.2, random_state=42. A stratified 70/30 split yields 140 training and 60 test samples; StandardScaler fits on training only. Both classifiers have 2 inputs, 8 ReLU hidden units and 1 sigmoid output (33 trainable parameters). They use identical Xavier-initialized weights, zero biases, mean BCE, full-batch SGD, learning rate=0.15 and 200 epochs. The test partition is used only for final accuracy and positive-class F1.

Training uses CPU float64 with one numerical-library thread. Separate 20-epoch warm-ups are discarded, followed by five fresh runs of each framework starting at the same initial weights. Run order alternates between scalar-first and PyTorch-first. Timed training includes gradient reset, forward calculation, loss, backward and parameter updates. Initialization, input conversion, initial/final evaluation and warm-up are excluded. This compares an educational scalar Python engine with vectorized native PyTorch; the speed difference includes tensor batching and implementation overhead, not only the differentiation algorithm.

The execution experiment uses `torch.jit.trace` followed by `torch.jit.freeze`, an allowed graph-execution alternative in the assignment. This choice does not claim that `torch.compile` is unavailable. The trained model runs in eval mode with `torch.no_grad()`. Tracing/checking and freezing preparation are timed separately. For batch sizes [1, 64], both modes are warmed up for 100 calls, then timed across 1000 calls per repeat; the reported value is the median of five amortized per-call times. All outputs are checked for float64 agreement. The same frozen graph is reused across batch sizes. Tracing records the operations executed for the example and does not validate arbitrary data-dependent Python control flow; this model has none.

The tensor-layout experiment compares identical values in a non-contiguous strided view `storage[:, ::2]` and a contiguous copy. Both inputs have shape [512, 512], dtype torch.float64; their strides are [1024, 2] and [512, 1]. The operation is `sum(dim=1)`. Each case has 30 warm-up calls and five repeats of 200 operations. A separate copy-then-sum case includes allocation and copying on every call. These CPU measurements do not require asynchronous GPU synchronization.

Run from the repository root with `./lab05/run.sh`. [config.yaml](config.yaml) stores all settings; [results/config_resolved.yaml](results/config_resolved.yaml) consolidates raw timings, initial weights, source IDs, graph metadata, environment versions and correctness checks. No external dataset download is required.

## 3. Results

### Gradient verification

| Comparison | Maximum absolute gradient difference | Requirement |
|---|---:|---:|
| Scalar engine vs PyTorch | 0.000e+00 | < 1e-6 |
| Scalar engine vs central differences | 1.152e-10 | < 1e-6 |
| PyTorch vs central differences | 1.152e-10 | < 1e-6 |

All 15 input gradients pass. The scalar verification graph contains 41 nodes. Individual inputs and derivatives are saved in [results/gradient_check.csv](results/gradient_check.csv). Float64 avoids making a 1e-6 comparison depend on float32 rounding.

### Matched classifier training

| Framework | Median training (ms), 5 runs | Initial BCE | Final BCE | Test accuracy | Test F1 |
|---|---:|---:|---:|---:|---:|
| scalar_autodiff | 7066.916 | 0.865339 | 0.320164 | 0.9333 | 0.9375 |
| pytorch | 12.137 | 0.865339 | 0.320164 | 0.9333 | 0.9375 |

PyTorch's median training time is 582.2 times lower on this host for this workload. Maximum differences across matched runs are 2.220e-16 for final parameters, 2.220e-16 for test probabilities and 6.661e-16 for training losses. Training paths are checked within absolute tolerance 1e-10. The same test split and initial weights are used throughout; no hyperparameters are selected from test results. Metrics are also saved in [results/training.csv](results/training.csv).

### Eager versus traced inference

| Batch | Eager (us/call) | Traced and frozen (us/call) | Eager / traced time | Estimated preparation break-even (calls) |
|---|---:|---:|---:|---:|
| 1 | 6.302 | 3.500 | 1.801 | 8003 |
| 64 | 8.508 | 5.236 | 1.625 | 6856 |

Tracing/checking took 22.060 ms and freezing took 0.370 ms, for 22.431 ms of one-time preparation, excluded from steady-state times. For batch 1, measured preparation is recovered after approximately 8,003 calls, using the difference between median per-call times. For batch 64, measured preparation is recovered after approximately 6,856 calls, using the difference between median per-call times. These estimates exclude other application startup costs and depend on host load. A ratio above 1 means traced execution was faster for that measured batch; a ratio below 1 means eager was faster.

### Tensor layout

| Variant | Median (us/call), 5 repeats | Copy cost |
|---|---:|---|
| noncontiguous_sum | 40.739 | excluded |
| contiguous_sum | 26.330 | excluded |
| copy_then_sum | 95.531 | included every call |

The non-contiguous/contiguous reduction-time ratio is 1.547. The contiguous input is prepared once before its reduction benchmark; the copy-then-sum result shows the complete cost when copying is repeated. Outputs agree within the recorded float64 tolerances. The observed direction and magnitude apply to this strided view and operation; not every non-contiguous tensor or kernel has the same penalty. Per-call timings and input properties are in [results/framework_benchmarks.csv](results/framework_benchmarks.csv).

The eager autograd graph contains 10 nodes with operations including AccumulateGrad, AddmmBackward0, ReluBackward0, SigmoidBackward0, TBackward0. The traced/frozen graph contains 8 IR nodes. [results/graphs.txt](results/graphs.txt) records the custom scalar graph, eager backward-function graph and frozen traced forward graph. They represent different graph levels and their node counts are not interchangeable performance measures.

Runtime: Python 3.11.15, PyTorch 2.3.1, macOS-27.0.1-arm64-arm-64bit, arm64; CPU threads=1. Timings are medians from this host, not device-independent guarantees. Correctness checks and deterministic training are reproducible under the recorded environment; wall-clock times will vary.

## 4. Three conclusions

1. The custom scalar engine constructs graphs and accumulates reverse-mode gradients correctly for the tested operations and shared graphs. For the required expression, its largest PyTorch gradient difference is 0.000e+00, below 1e-6; central differences independently confirm the gradients.
2. With matching data, initialization, architecture and SGD updates, both engines reach BCE 0.320164 and test accuracy 0.9333 to the recorded tolerance. PyTorch trains 582.2 times faster in this experiment because tensor batching and native operations avoid much of the scalar Python graph overhead. This is a workload-specific comparison.
3. Graph execution can help when repeated inference saves enough per-call overhead to recover preparation costs; the measured batch-specific ratios and break-even estimates show when that happened here. Memory layout also matters: compare the measured reduction and copy-then-sum times before deciding to make a contiguous copy. Compilation or copying should be assessed against the actual workload and reuse pattern.

Sources: [PyTorch 2.3 autograd](https://docs.pytorch.org/docs/2.3/autograd.html), [torch.jit.trace](https://docs.pytorch.org/docs/2.3/generated/torch.jit.trace.html), [BCE](https://docs.pytorch.org/docs/2.3/generated/torch.nn.BCELoss.html), [make_moons](https://scikit-learn.org/1.5/modules/generated/sklearn.datasets.make_moons.html).
