#!/usr/bin/env python3
"""Run Lab 2: NumPy neural-network training, gradient checks and pass timing."""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_key] = '1'

import argparse
import hashlib
import importlib.metadata
import platform
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ml-vistula-lab02-matplotlib'))
if 'XDG_CACHE_HOME' not in os.environ:
    _font_cache = Path(tempfile.gettempdir()) / 'ml-vistula-lab02-cache'
    _font_cache.mkdir(parents=True, exist_ok=True)
    os.environ['XDG_CACHE_HOME'] = str(_font_cache)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from lab02.src.experiment import prepare_data, train_model
from lab02.src.gradient_check import check_gradients
from lab02.src.network import NumpyMLP
from lab02.src.timing import measure_passes


def validate_config(config):
    fixed = {
        ('data', 'name'): 'load_digits', ('data', 'scale_divisor'): 16.0,
        ('data', 'test_size'): 0.30, ('data', 'random_state'): 42,
        ('model', 'inputs'): 64, ('model', 'outputs'): 10,
        ('model', 'hidden_widths'): [16, 64], ('model', 'weight_mean'): 0.0,
        ('model', 'weight_std'): 0.01, ('model', 'bias_initialization'): 'zero',
        ('model', 'dtype'): 'float64', ('training', 'learning_rates'): [0.001, 0.01, 0.1],
        ('training', 'batch_size'): 32, ('training', 'epochs'): 50,
        ('gradient_check', 'samples'): 10, ('gradient_check', 'epsilon'): 1e-5,
        ('gradient_check', 'tolerance'): 1e-6, ('gradient_check', 'denominator_floor'): 1e-12,
        ('gradient_check', 'minimum_passes'): 9,
        ('timing', 'batch_size'): 32, ('timing', 'repeats'): 100,
        ('checks', 'probability_sum_tolerance'): 1e-6,
    }
    for (section, key), value in fixed.items():
        if config[section][key] != value:
            raise ValueError(f'{section}.{key} must be {value!r} for the specified experiment.')
    if not 1 <= config['gradient_check']['batch_size'] <= 1257:
        raise ValueError('Gradient-check batch size must fit the training partition.')
    if config['timing']['warmup'] < 1:
        raise ValueError('Pass timing requires a warm-up.')


def loss_plot(results, metrics, epochs):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
                         'axes.labelsize': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, layout='constrained')
    figure.suptitle('Digits classification: training cross-entropy', fontsize=15, fontweight='bold')
    colors = {0.001: '#4263EB', 0.01: '#D97706', 0.1: '#0F766E'}
    for axis, width in zip(axes, [16, 64]):
        for learning_rate in [0.001, 0.01, 0.1]:
            frame = epochs[(epochs['width'] == width) & (epochs['learning_rate'] == learning_rate)]
            initial = next(row['initial_loss'] for row in metrics if row['width'] == width and row['learning_rate'] == learning_rate)
            axis.plot([0, *frame['epoch'].tolist()], [initial, *frame['train_loss'].tolist()],
                      color=colors[learning_rate], linewidth=2.1,
                      linestyle='--' if learning_rate == 0.01 else '-', label=f'Learning rate {learning_rate:g}')
        axis.axhline(np.log(10), color='#94A3B8', linestyle='--', linewidth=1.0, label='ln(10) reference')
        count = next(row['parameters'] for row in metrics if row['width'] == width)
        axis.set_title(f'Hidden width W = {width} ({count:,} parameters)')
        axis.set_xlabel('Epoch')
        axis.set_xlim(0, 50)
        axis.set_ylim(bottom=0, top=2.43)
        axis.grid(axis='y', alpha=0.2)
        axis.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=8.5)
    axes[0].set_ylabel('Mean cross-entropy')
    figure.savefig(results / 'loss_curves.png', dpi=180, facecolor='white')
    plt.close(figure)


def write_report(config, data, metrics, gradient_details, runs, environment):
    gradient_table = '\n'.join(
        f"| {width} | {check['passed_weights']}/10 | {check['both_zero_count']} | {check['max_relative_error']:.3e} | {check['max_absolute_error']:.3e} | {'PASS' if check['passed'] else 'FAIL'} |"
        for width, check in gradient_details.items())
    metric_table = '\n'.join(
        f"| {row['width']} | {row['learning_rate']:g} | {row['parameters']} | {row['initial_loss']:.6f} | {row['final_train_loss']:.6f} | {row['final_train_accuracy']:.4f} | {row['test_accuracy']:.4f} |"
        for row in metrics)
    timing_table = '\n'.join(
        f"| {row['width']} | {row['learning_rate']:g} | {1000 * row['median_forward_ms']:.3f} | {1000 * row['median_backward_ms']:.3f} | {row['median_backward_ms'] / row['median_forward_ms']:.3f} |"
        for row in metrics)
    flops_table = '\n'.join(
        f"| {width} | {next(run['timing']['matrix_multiplication_flops']['forward'] for run in runs if run['width'] == width):,} | {next(run['timing']['matrix_multiplication_flops']['backward'] for run in runs if run['width'] == width):,} |"
        for width in config['model']['hidden_widths'])
    best = max(metrics, key=lambda row: row['test_accuracy'])
    backward_dominates = sum(row['median_backward_ms'] > row['median_forward_ms'] for row in metrics)
    shape_error = max(run['max_probability_row_sum_error'] for run in runs)
    nonzero_failures = sum(10 - check['passed_weights'] for check in gradient_details.values())
    text = f"""# Lab 2: A neural network from scratch

## 1. Goal

Design and implement a two-hidden-layer classifier using explicit NumPy forward propagation and chain-rule backpropagation. Verify its softmax probabilities, cross-entropy and analytical gradients, compare the required learning rates and hidden widths on handwritten digits, and measure forward/backward latency separately.

## 2. Method

The built-in `load_digits` dataset contains {data['rows']:,} samples: flattened 8-by-8 images with 64 pixel inputs and labels 0 through 9. Inputs are divided by 16, yielding values in [0, 1]; labels are encoded as 10-column one-hot vectors. A stratified 70/30 split with random_state=42 gives {data['train_rows']:,} training and {data['test_rows']:,} test examples. Pixel scaling is a fixed rule and learns no test-set statistics. The test partition is used only for final evaluation.

### Network design and dimensions

Both hidden layers use the same width W, tested at 16 and 64:

```text
64 inputs -> affine -> W -> ReLU -> affine -> W -> ReLU -> affine -> 10 -> softmax
```

| Parameter | General shape | W = 16 | W = 64 |
|---|---|---|---|
| W1 | (64, W) | (64, 16) | (64, 64) |
| b1 | (W,) | (16,) | (64,) |
| W2 | (W, W) | (16, 16) | (64, 64) |
| b2 | (W,) | (16,) | (64,) |
| W3 | (W, 10) | (16, 10) | (64, 10) |
| b3 | (10,) | (10,) | (10,) |
| Trainable scalars | 64W + W^2 + 12W + 10 | 1,482 | 8,970 |

Weights use independent normal draws with mean 0 and standard deviation 0.01: `rng.normal(0, 0.01, shape)`. This explicitly states the NumPy scale convention used for the manual's N(0, 0.01). Biases start at zero. The initialization seed is {config['model']['initialization_seed']}. All calculations use float64. For a fixed width, all three learning rates start from identical parameters and use the same sequence of shuffled training rows.

### Forward pass, loss and analytical derivation

For a batch of m row vectors, the layer equations are:

```text
Z1 = X @ W1 + b1             A1 = max(Z1, 0)
Z2 = A1 @ W2 + b2            A2 = max(Z2, 0)
Z3 = A2 @ W3 + b3            P  = softmax(Z3)
```

Biases broadcast across rows. Softmax subtracts each row's maximum before exponentiation. Its output has shape (m, 10), and each row must sum to 1 within 1e-6. Mean cross-entropy is `L = -(1/m) * sum_i sum_k Y[i,k] * log(P[i,k])`. It is evaluated using stable log-sum-exp from logits, preserving finite losses without clipping probabilities. Uniform probabilities give `L = ln(10) = 2.302585...`; the randomly initialized network should be close to that value.

To derive the output gradient, softmax has derivative `dP[j]/dZ[k] = P[j] * (delta[j,k] - P[k])`, while cross-entropy contributes `dL/dP[j] = -Y[j]/(m*P[j])`. Summing their chain-rule products and using `sum_j Y[j] = 1` gives `dZ3 = (P - Y)/m`. Applying the affine-layer rule and the ReLU indicator at each preceding layer gives:

```text
dZ3 = (P - Y) / m
dW3 = A2.T @ dZ3            db3 = sum(dZ3, axis=0)
dZ2 = (dZ3 @ W3.T) * (Z2 > 0)
dW2 = A1.T @ dZ2            db2 = sum(dZ2, axis=0)
dZ1 = (dZ2 @ W2.T) * (Z1 > 0)
dW1 = X.T @ dZ1             db1 = sum(dZ1, axis=0)
```

Every gradient matches its parameter's shape. ReLU's derivative is zero at zero. The normalization by m occurs once, at dZ3, and uses the actual batch size, including the final batch of 9 records. SGD applies `parameter -= learning_rate * gradient`. No automatic differentiation or neural-network training library is used.

For each width, numerical checking uses the first {config['gradient_check']['batch_size']} training rows and 10 weights drawn uniformly without replacement from the union of W1, W2 and W3 with random_state={config['gradient_check']['random_state']}. Biases are excluded from this required random-weight sample and are covered by separate unit tests. Each selected weight is perturbed by +1e-5 and -1e-5, giving `(L(w+h)-L(w-h))/(2h)`. Relative error is `abs(analytic-numerical)/max(1e-12, abs(analytic)+abs(numerical))`; a weight passes only if this is strictly below 1e-6. At least 9/10 must pass for each width. Indices are selected before inspecting derivatives, all comparisons are retained, and parameters are restored after checking.

Training uses shuffled mini-batches of 32 for 50 epochs, learning rates [0.001, 0.01, 0.1], and widths [16, 64]. Each epoch has 40 updates (39 batches of 32 and one of 9). The shuffle generator resets to seed {config['training']['shuffle_seed']} for each configuration. Full-training loss and accuracy are computed after each epoch, producing 300 logged rows. The test set is not used for epoch logging, stopping or hyperparameter selection; all six predefined test results are reported.

Pass timings use the trained model and a fixed batch of 32 training examples. Forward and backward each have {config['timing']['warmup']} warm-up calls followed by 100 individually timed calls; medians use `perf_counter_ns`. Forward includes activations, softmax and cache construction. Backward consumes one precomputed forward cache and computes parameter gradients; another forward pass, cross-entropy, SGD updates, setup and benchmark-wrapper validation are excluded. Model-internal input and finite-value checks remain included. CPU numerical-library threads are limited to one. These are complete NumPy-function latencies, not isolated matrix-kernel timings.

Run from the repository root with `./lab02/run.sh`. [config.yaml](config.yaml) stores the experiment settings; [results/config_resolved.yaml](results/config_resolved.yaml) includes versions, split IDs, initialization/final hashes, all raw pass timings and correctness checks.

## 3. Results

### Gradient verification and sanity checks

| Width | Weights below 1e-6 relative error | Both derivatives exactly zero | Maximum relative error, all 10 | Maximum absolute error | Requirement |
|---|---:|---:|---:|---:|---|
{gradient_table}

Both widths meet the required 9/10 threshold. {nonzero_failures} comparisons in total exceed the relative-error threshold; their absolute errors and small derivatives are preserved in [results/gradient_check.csv](results/gradient_check.csv). Finite differences of nearly equal losses can lose relative precision for tiny derivatives. Exactly-zero checks are counted explicitly rather than hidden; separate tests also validate nonzero gradients of all six parameter types. The largest observed probability row-sum error is {shape_error:.3e}, below 1e-6. Initial losses in the next table are close to ln(10).

### Learning-rate and width comparison

| Width | Learning rate | Parameters | Initial train loss | Final train loss | Final train accuracy | Final test accuracy |
|---|---:|---:|---:|---:|---:|---:|
{metric_table}

These six rows follow the specified 50-epoch budget and initialization scale. The highest observed test accuracy is {best['test_accuracy']:.4f} at W={best['width']}, learning rate={best['learning_rate']:g}; this is a descriptive comparison on one holdout, not a model-selection protocol or a guarantee on other splits. Detailed metrics are in [results/metrics.csv](results/metrics.csv).

At learning rates 0.001 and 0.01, both widths remain close to ln(10) loss and chance-level accuracy within this budget. Small weights in two hidden layers produce small early gradients, so these settings make little progress in 50 epochs. This does not establish that those rates always fail. At learning rate 0.1, the wider network starts improving earlier and reaches a lower final training loss in this experiment. Mini-batch SGD can produce non-monotonic epoch losses, visible in the higher-rate curves.

![Training loss for the six required configurations](results/loss_curves.png)

The curves show full-training cross-entropy after each epoch; epoch 0 adds the measured initial loss. A gray dashed line marks ln(10). The two panels share a loss scale, and learning-rate colors are consistent. The two low-rate curves nearly overlap the reference at this scale; full-precision values remain in [results/epoch_log.csv](results/epoch_log.csv), containing all 300 post-epoch measurements.

### Forward and backward timing

| Width | Learning rate | Forward median (us/batch) | Backward median (us/batch) | Backward / forward time |
|---|---:|---:|---:|---:|
{timing_table}

Backward takes longer in {backward_dominates}/6 measured configurations. Forward evaluates three dense products; backward evaluates five, including weight gradients and the two hidden activation gradients. The unused input gradient dX is not computed. Counting one multiply-add as two floating-point operations gives these leading matrix-product estimates for batch 32:

| Width | Forward matrix-product FLOPs | Backward matrix-product FLOPs |
|---|---:|---:|
{flops_table}

The forward estimate is `2m*(64W + W^2 + 10W)`; backward is `2m*(64W + 2W^2 + 20W)`. Extra matrix products help explain backward cost, but the measured ratio also includes activations, reductions, allocation, NumPy dispatch and model validation. Kernel efficiency, dimensions and host load matter; FLOP counts alone do not determine the timing order. All 100 raw durations per pass and configuration are saved in the YAML evidence. Runtime: Python {environment['python']}, NumPy {environment['packages']['numpy']}, {environment['platform']}, {environment['architecture']}. Timings will vary between runs and machines.

## 4. Three conclusions

1. The NumPy implementation has dimensionally correct forward and backward passes, stable softmax/cross-entropy, and probability sums within 1e-6. Both hidden widths pass the required random-weight gradient check, with all sampled errors and zero derivatives reported.
2. Learning rate and hidden width affect progress within the fixed 50-epoch budget. The highest observed test accuracy is {best['test_accuracy']:.4f} for W={best['width']} and learning rate {best['learning_rate']:g}; the loss curves show how all six predefined settings trained. A broader tuning study would require a separate validation protocol.
3. Backward is slower in {backward_dominates}/6 measured settings. Its five matrix products versus three in forward increase computational work, while validation, allocation and small-matrix overhead also affect the complete-function latency. The saved 100-run medians and raw timings make that comparison inspectable.

Sources: [load_digits](https://scikit-learn.org/1.5/modules/generated/sklearn.datasets.load_digits.html), [NumPy normal initialization](https://numpy.org/doc/1.26/reference/random/generated/numpy.random.Generator.normal.html).
"""
    (ROOT / 'report.md').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    validate_config(config)
    results = ROOT / config['paths']['results']
    results.mkdir(parents=True, exist_ok=True)
    packages = [line.split('==')[0] for line in (ROOT / 'requirements.txt').read_text().splitlines() if '==' in line]
    environment = {'python': platform.python_version(), 'platform': platform.platform(),
                   'architecture': platform.machine(), 'logical_cpus': os.cpu_count(),
                   'numerical_threads': 1, 'threadpools': threadpool_info(),
                   'packages': {name: importlib.metadata.version(name) for name in packages}}
    data, data_info = prepare_data(config['data'])
    print(f"Lab 2: load_digits {data_info['rows']} x 64; train={data_info['train_rows']}, test={data_info['test_rows']}", flush=True)
    gradient_rows, gradient_details = [], {}
    check_config = config['gradient_check']
    for width in config['model']['hidden_widths']:
        model = NumpyMLP(width, seed=config['model']['initialization_seed'], weight_std=config['model']['weight_std'])
        batch = check_config['batch_size']
        rows, detail = check_gradients(model, data['train_x'][:batch], data['train_y'][:batch],
                                       seed=check_config['random_state'], samples=check_config['samples'],
                                       epsilon=check_config['epsilon'], tolerance=check_config['tolerance'])
        gradient_rows.extend({'width': width, **row} for row in rows)
        gradient_details[width] = detail
        print(f"Gradient check W={width}: {detail['passed_weights']}/10 PASS (<1e-6); max absolute error={detail['max_absolute_error']:.3e}", flush=True)
    pd.DataFrame(gradient_rows).to_csv(results / 'gradient_check.csv', index=False, float_format='%.17g')
    if not all(detail['passed'] for detail in gradient_details.values()):
        raise AssertionError('Gradient check failed; inspect results/gradient_check.csv.')
    metrics, epoch_rows, runs = [], [], []
    initial_hashes = {}
    for width in config['model']['hidden_widths']:
        for learning_rate in config['training']['learning_rates']:
            model, row, history, detail = train_model(data, width, learning_rate, config)
            previous = initial_hashes.setdefault(width, detail['initial_parameters_sha256'])
            if previous != detail['initial_parameters_sha256']:
                raise AssertionError('Learning-rate runs did not start from identical parameters.')
            timing_batch = config['timing']['batch_size']
            timing = measure_passes(model, data['train_x'][:timing_batch], data['train_y'][:timing_batch],
                                    repeats=config['timing']['repeats'], warmup=config['timing']['warmup'])
            row.update({'median_forward_ms': timing['median_forward_ms'], 'median_backward_ms': timing['median_backward_ms']})
            detail['timing'] = timing
            detail['metrics'] = row
            metrics.append(row)
            epoch_rows.extend(history)
            runs.append(detail)
            print(f"RESULT W={width} lr={learning_rate:g}: test_accuracy={row['test_accuracy']:.4f}; forward={row['median_forward_ms']:.6f}ms; backward={row['median_backward_ms']:.6f}ms", flush=True)
    pd.DataFrame(metrics).to_csv(results / 'metrics.csv', index=False, float_format='%.17g')
    epochs = pd.DataFrame(epoch_rows)
    epochs.to_csv(results / 'epoch_log.csv', index=False, float_format='%.17g')
    loss_plot(results, metrics, epochs)
    manifest = {'config': config, 'environment': environment, 'data': data_info,
                'gradient_checks': gradient_details, 'runs': runs,
                'artifact_sha256': {name: hashlib.sha256((results / name).read_bytes()).hexdigest()
                                    for name in ['metrics.csv', 'epoch_log.csv', 'gradient_check.csv', 'loss_curves.png']}}
    (results / 'config_resolved.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False), encoding='utf-8')
    write_report(config, data_info, metrics, gradient_details, runs, environment)
    best = max(metrics, key=lambda row: row['test_accuracy'])
    print(f"Completed 6 configurations, 300 epoch rows, and 100 forward/backward timings per model.\nHighest observed test accuracy: {best['test_accuracy']:.4f} (W={best['width']}, lr={best['learning_rate']:g}).\nEvidence: {results}\nReport: {ROOT / 'report.md'}", flush=True)


if __name__ == '__main__':
    main()
