#!/usr/bin/env python3
"""Run Lab 5: scalar autodiff, matched training, execution and tensor layouts."""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_key] = '1'

import argparse
import hashlib
import importlib.metadata
import math
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from lab05.src.framework_benchmarks import run_framework_benchmarks
from lab05.src.training import run_training
from lab05.src.verification import verify_gradients


def validate_config(config):
    required = {'inputs': 2, 'hidden': 8, 'outputs': 1, 'hidden_activation': 'relu',
                'output_activation': 'sigmoid', 'loss': 'binary_cross_entropy', 'dtype': 'float64'}
    for key, value in required.items():
        if config['model'][key] != value:
            raise ValueError(f'model.{key} must be {value!r} for the required experiment.')
    if config['data']['test_size'] != 0.30:
        raise ValueError('data.test_size must be 0.30 for the documented 70/30 experiment.')
    for section in ['training', 'execution', 'tensors']:
        if config[section]['repeats'] != 5:
            raise ValueError(f'{section}.repeats must be 5.')
    if config['training']['epochs'] < 1 or config['training']['warmup_epochs'] < 1:
        raise ValueError('Training and warm-up epochs must be positive.')
    if not math.isfinite(config['training']['learning_rate']) or config['training']['learning_rate'] <= 0:
        raise ValueError('Learning rate must be finite and positive.')
    gradient = config['gradient_check']
    if not 0 < gradient['epsilon'] < 0.01 or gradient['tolerance'] != 1e-6 or gradient['elements'] < 1:
        raise ValueError('Gradient check needs positive epsilon <0.01, elements >=1 and tolerance=1e-6.')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(config, gradients, training_rows, training, frameworks, manifest):
    checks = gradients['max_errors']
    scalar, native = training_rows
    speedup = scalar['median_training_ms'] / native['median_training_ms']
    training_table = '\n'.join(
        f"| {row['framework']} | {row['median_training_ms']:.3f} | {row['initial_bce']:.6f} | {row['final_bce']:.6f} | {row['test_accuracy']:.4f} | {row['test_f1']:.4f} |"
        for row in training_rows)
    execution = frameworks['execution']
    execution_table = '\n'.join(
        f"| {batch['batch_size']} | {1000 * batch['eager']['median_ms']:.3f} | {1000 * batch['torchscript_trace_frozen']['median_ms']:.3f} | {batch['eager_over_traced_speedup']:.3f} | {batch['break_even_calls'] if batch['break_even_calls'] is not None else 'No measured break-even'} |"
        for batch in execution['batches'])
    tensors = frameworks['tensors']
    tensor_table = '\n'.join(
        f"| {variant} | {1000 * result['median_ms']:.3f} | {'included every call' if variant == 'copy_then_sum' else 'excluded'} |"
        for variant, result in tensors['measurements'].items())
    compilation_notes = []
    for batch in execution['batches']:
        if batch['break_even_calls'] is None:
            compilation_notes.append(f"For batch {batch['batch_size']}, tracing did not reduce measured steady-state latency, so no preparation break-even is estimated.")
        else:
            compilation_notes.append(f"For batch {batch['batch_size']}, measured preparation is recovered after approximately {batch['break_even_calls']:,} calls, using the difference between median per-call times.")
    graph_ops = ', '.join(sorted(set(node['operation'] for node in frameworks['graphs']['autograd']['nodes'])))
    text = f"""# Lab 5: Frameworks and automatic differentiation

## 1. Goal

Implement a scalar automatic-differentiation engine that constructs computation graphs and performs reverse-mode differentiation with correct gradient accumulation. Validate derivatives against PyTorch and numerical estimates, train the same small classifier in both engines, and measure eager versus traced execution and the effect of tensor contiguity.

## 2. Method

The custom `Value` node stores a scalar value, gradient, ordered parent references, operation name and local derivative rule. Addition, multiplication, constant powers and ReLU construct graph nodes during the forward pass. Subtraction, division, exponentials, logarithms and sigmoid support the classifier and binary cross-entropy (BCE). The backward pass obtains an iterative topological ordering, resets reachable gradients, seeds the scalar output gradient with 1, and traverses in reverse while adding each chain-rule contribution to parent gradients. Repeated operands and shared intermediate nodes accumulate their contributions. ReLU uses derivative zero at zero. A repeated `backward()` starts a fresh gradient calculation; this differs from PyTorch leaf-gradient accumulation across separate calls. Local derivative closures avoid output-reference cycles.

The required expression `((a*b + c)**2).relu().sum()` is evaluated using {config['gradient_check']['elements']} elements per input vector. Python scalar nodes are summed to obtain the same scalar objective as PyTorch float64 tensors. Every a, b and c input derivative is compared with PyTorch autograd and central finite differences using epsilon={config['gradient_check']['epsilon']:g}. Each maximum absolute error must be strictly below {config['gradient_check']['tolerance']:g}. Additional tests cover shared branches, repeated operands, powers, ReLU, domain errors and deep graphs.

The generated `make_moons` dataset contains {training['data']['rows']} samples with noise={config['data']['noise']}, random_state={config['data']['random_state']}. A stratified 70/30 split yields {training['data']['train_rows']} training and {training['data']['test_rows']} test samples; StandardScaler fits on training only. Both classifiers have 2 inputs, 8 ReLU hidden units and 1 sigmoid output ({scalar['parameters']} trainable parameters). They use identical Xavier-initialized weights, zero biases, mean BCE, full-batch SGD, learning rate={config['training']['learning_rate']} and {config['training']['epochs']} epochs. The test partition is used only for final accuracy and positive-class F1.

Training uses CPU float64 with one numerical-library thread. Separate {config['training']['warmup_epochs']}-epoch warm-ups are discarded, followed by five fresh runs of each framework starting at the same initial weights. Run order alternates between scalar-first and PyTorch-first. Timed training includes gradient reset, forward calculation, loss, backward and parameter updates. Initialization, input conversion, initial/final evaluation and warm-up are excluded. This compares an educational scalar Python engine with vectorized native PyTorch; the speed difference includes tensor batching and implementation overhead, not only the differentiation algorithm.

The execution experiment uses `torch.jit.trace` followed by `torch.jit.freeze`, an allowed graph-execution alternative in the assignment. This choice does not claim that `torch.compile` is unavailable. The trained model runs in eval mode with `torch.no_grad()`. Tracing/checking and freezing preparation are timed separately. For batch sizes {config['execution']['batches']}, both modes are warmed up for {config['execution']['warmup']} calls, then timed across {config['execution']['iterations']} calls per repeat; the reported value is the median of five amortized per-call times. All outputs are checked for float64 agreement. The same frozen graph is reused across batch sizes. Tracing records the operations executed for the example and does not validate arbitrary data-dependent Python control flow; this model has none.

The tensor-layout experiment compares identical values in a non-contiguous strided view `storage[:, ::2]` and a contiguous copy. Both inputs have shape {tensors['noncontiguous']['shape']}, dtype {tensors['noncontiguous']['dtype']}; their strides are {tensors['noncontiguous']['stride']} and {tensors['contiguous']['stride']}. The operation is `sum(dim=1)`. Each case has {config['tensors']['warmup']} warm-up calls and five repeats of {config['tensors']['iterations']} operations. A separate copy-then-sum case includes allocation and copying on every call. These CPU measurements do not require asynchronous GPU synchronization.

Run from the repository root with `./lab05/run.sh`. [config.yaml](config.yaml) stores all settings; [results/config_resolved.yaml](results/config_resolved.yaml) consolidates raw timings, initial weights, source IDs, graph metadata, environment versions and correctness checks. No external dataset download is required.

## 3. Results

### Gradient verification

| Comparison | Maximum absolute gradient difference | Requirement |
|---|---:|---:|
| Scalar engine vs PyTorch | {checks['abs_scalar_pytorch']:.3e} | < 1e-6 |
| Scalar engine vs central differences | {checks['abs_scalar_numerical']:.3e} | < 1e-6 |
| PyTorch vs central differences | {checks['abs_pytorch_numerical']:.3e} | < 1e-6 |

All {gradients['compared_input_gradients']} input gradients pass. The scalar verification graph contains {gradients['scalar_graph_nodes']} nodes. Individual inputs and derivatives are saved in [results/gradient_check.csv](results/gradient_check.csv). Float64 avoids making a 1e-6 comparison depend on float32 rounding.

### Matched classifier training

| Framework | Median training (ms), 5 runs | Initial BCE | Final BCE | Test accuracy | Test F1 |
|---|---:|---:|---:|---:|---:|
{training_table}

PyTorch's median training time is {speedup:.1f} times lower on this host for this workload. Maximum differences across matched runs are {training['max_parameter_difference']:.3e} for final parameters, {training['max_test_probability_difference']:.3e} for test probabilities and {training['max_training_loss_difference']:.3e} for training losses. Training paths are checked within absolute tolerance 1e-10. The same test split and initial weights are used throughout; no hyperparameters are selected from test results. Metrics are also saved in [results/training.csv](results/training.csv).

### Eager versus traced inference

| Batch | Eager (us/call) | Traced and frozen (us/call) | Eager / traced time | Estimated preparation break-even (calls) |
|---|---:|---:|---:|---:|
{execution_table}

Tracing/checking took {execution['trace_ms']:.3f} ms and freezing took {execution['freeze_ms']:.3f} ms, for {execution['preparation_ms']:.3f} ms of one-time preparation, excluded from steady-state times. {' '.join(compilation_notes)} These estimates exclude other application startup costs and depend on host load. A ratio above 1 means traced execution was faster for that measured batch; a ratio below 1 means eager was faster.

### Tensor layout

| Variant | Median (us/call), 5 repeats | Copy cost |
|---|---:|---|
{tensor_table}

The non-contiguous/contiguous reduction-time ratio is {tensors['noncontiguous_over_contiguous_speedup']:.3f}. The contiguous input is prepared once before its reduction benchmark; the copy-then-sum result shows the complete cost when copying is repeated. Outputs agree within the recorded float64 tolerances. The observed direction and magnitude apply to this strided view and operation; not every non-contiguous tensor or kernel has the same penalty. Per-call timings and input properties are in [results/framework_benchmarks.csv](results/framework_benchmarks.csv).

The eager autograd graph contains {len(frameworks['graphs']['autograd']['nodes'])} nodes with operations including {graph_ops}. The traced/frozen graph contains {frameworks['graphs']['traced_frozen']['node_count']} IR nodes. [results/graphs.txt](results/graphs.txt) records the custom scalar graph, eager backward-function graph and frozen traced forward graph. They represent different graph levels and their node counts are not interchangeable performance measures.

Runtime: Python {manifest['environment']['python']}, PyTorch {manifest['environment']['packages']['torch']}, {manifest['environment']['platform']}, {manifest['environment']['architecture']}; CPU threads={manifest['environment']['torch_threads']}. Timings are medians from this host, not device-independent guarantees. Correctness checks and deterministic training are reproducible under the recorded environment; wall-clock times will vary.

## 4. Three conclusions

1. The custom scalar engine constructs graphs and accumulates reverse-mode gradients correctly for the tested operations and shared graphs. For the required expression, its largest PyTorch gradient difference is {checks['abs_scalar_pytorch']:.3e}, below 1e-6; central differences independently confirm the gradients.
2. With matching data, initialization, architecture and SGD updates, both engines reach BCE {scalar['final_bce']:.6f} and test accuracy {scalar['test_accuracy']:.4f} to the recorded tolerance. PyTorch trains {speedup:.1f} times faster in this experiment because tensor batching and native operations avoid much of the scalar Python graph overhead. This is a workload-specific comparison.
3. Graph execution can help when repeated inference saves enough per-call overhead to recover preparation costs; the measured batch-specific ratios and break-even estimates show when that happened here. Memory layout also matters: compare the measured reduction and copy-then-sum times before deciding to make a contiguous copy. Compilation or copying should be assessed against the actual workload and reuse pattern.

Sources: [PyTorch 2.3 autograd](https://docs.pytorch.org/docs/2.3/autograd.html), [torch.jit.trace](https://docs.pytorch.org/docs/2.3/generated/torch.jit.trace.html), [BCE](https://docs.pytorch.org/docs/2.3/generated/torch.nn.BCELoss.html), [make_moons](https://scikit-learn.org/1.5/modules/generated/sklearn.datasets.make_moons.html).
"""
    (ROOT / 'report.md').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    validate_config(config)
    package_names = [line.split('==')[0] for line in (ROOT / 'requirements.txt').read_text().splitlines() if '==' in line]
    package_versions = {name: importlib.metadata.version(name) for name in package_names}
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    results = ROOT / config['paths']['results']
    results.mkdir(parents=True, exist_ok=True)
    print('Lab 5: verifying scalar autodiff against PyTorch and numerical gradients...', flush=True)
    gradient_rows, gradients, scalar_graph = verify_gradients(config['gradient_check'])
    print(f"Gradient check PASS: max scalar/PyTorch={gradients['max_errors']['abs_scalar_pytorch']:.3e}; scalar/numerical={gradients['max_errors']['abs_scalar_numerical']:.3e}", flush=True)
    pd.DataFrame(gradient_rows).to_csv(results / 'gradient_check.csv', index=False, float_format='%.17g')
    print('Training matched 2 -> 8 -> 1 classifiers; 5 fresh runs per engine...', flush=True)
    training_rows, training, model = run_training(config)
    pd.DataFrame(training_rows).to_csv(results / 'training.csv', index=False, float_format='%.17g')
    print('Benchmarking warmed-up eager/traced execution and tensor layouts...', flush=True)
    framework_rows, frameworks, framework_graphs = run_framework_benchmarks(model, config)
    pd.DataFrame(framework_rows).to_csv(results / 'framework_benchmarks.csv', index=False, float_format='%.17g')
    (results / 'graphs.txt').write_text((scalar_graph + '\n' + framework_graphs).rstrip() + '\n', encoding='utf-8')
    manifest = {
        'config': config,
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'architecture': platform.machine(), 'processor': platform.processor(),
                        'logical_cpus': os.cpu_count(), 'device': 'cpu',
                        'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
                        'packages': package_versions},
        'gradient_check': gradients, 'training': training, 'framework_benchmarks': frameworks,
        'artifacts_sha256': {name: sha256(results / name) for name in ['gradient_check.csv', 'training.csv', 'framework_benchmarks.csv', 'graphs.txt']},
    }
    (results / 'config_resolved.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False), encoding='utf-8')
    write_report(config, gradients, training_rows, training, frameworks, manifest)
    scalar, native = training_rows
    print(f"Median training: scalar={scalar['median_training_ms']:.3f} ms; PyTorch={native['median_training_ms']:.3f} ms", flush=True)
    print(f"Test accuracy={scalar['test_accuracy']:.4f}; F1={scalar['test_f1']:.4f}; BCE={scalar['final_bce']:.6f}", flush=True)
    for batch in frameworks['execution']['batches']:
        print(f"Batch {batch['batch_size']}: eager={1000 * batch['eager']['median_ms']:.3f} us; traced={1000 * batch['torchscript_trace_frozen']['median_ms']:.3f} us", flush=True)
    print(f'Evidence: {results}\nReport: {ROOT / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
