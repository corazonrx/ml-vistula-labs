# Lab 2: A neural network from scratch

This lab implements a two-hidden-layer classifier and its backpropagation in NumPy, checks the gradients numerically, and compares six training configurations. [report.md](report.md) contains the Goal, Method, Results, and Three conclusions, including the matrix shapes and chain-rule derivation.

## Run

From the repository root:

```bash
./lab02/run.sh
```

The script creates `lab02/.venv`, installs the pinned dependencies in `lab02/requirements.txt`, and runs the experiment using Python 3.11. The first installation needs internet access. The digits dataset is bundled with scikit-learn and needs no separate download.

With the environment already installed:

```bash
lab02/.venv/bin/python lab02/run.py
lab02/.venv/bin/python lab02/run.py --config lab02/config.yaml
```

Optional focused tests:

```bash
lab02/.venv/bin/python -m pip install -r lab02/requirements-dev.txt
lab02/.venv/bin/python -m pytest lab02/tests -q
```

## Experiment

`load_digits` provides 1,797 handwritten digit images, each flattened into 64 inputs. Pixel values are divided by the known maximum, 16, and labels are encoded as 10-element one-hot vectors. A stratified 70/30 split with seed 42 gives 1,257 training and 540 test records. The network and its gradients use NumPy; scikit-learn supplies the dataset and split.

The architecture is `64 → W ReLU → W ReLU → 10 softmax`, with hidden width `W` equal to 16 or 64. Both hidden layers have the same width. Weights are drawn from a normal distribution with mean 0 and **standard deviation 0.01**; biases start at zero. The output must have shape `(batch_size, 10)` and each probability row must sum to one within tolerance `1e-6`. Uniform class probabilities give the cross-entropy sanity baseline `ln(10) ≈ 2.302585`; small random initial weights should produce a loss near that value.

Backpropagation is derived explicitly and implemented without an automatic-differentiation framework. Before training, 10 randomly selected weights per width are checked with central finite differences using perturbations `±1e-5`. The target is relative error strictly below `1e-6` for at least 9 of the 10 weights in each check. The CSV preserves all 20 comparisons.

Each width is trained with learning rates `0.001`, `0.01`, and `0.1`: six configurations in total. Training uses shuffled mini-batches of 32 for 50 epochs. Within a width, all three learning rates start from identical weights and use the same seeded shuffle order. Full-training-set cross-entropy is evaluated after every epoch, giving 300 logged records. Test accuracy is evaluated after training; test results do not select hyperparameters or stopping points.

Forward and backward timings use a batch of 32, with 20 warm-up calls followed by 100 measured calls for each operation. Reported timings are medians; raw measurements and their scope are retained in the resolved configuration. The experiment records the measured direction of the difference between forward and backward time.

## Saved evidence

A normal run writes five files in `lab02/results/`:

| File | Contents |
| --- | --- |
| `metrics.csv` | Final test accuracy, training loss, and forward/backward timing summaries for all six configurations |
| `epoch_log.csv` | Full-training-set loss after every epoch: 6 configurations × 50 epochs |
| `gradient_check.csv` | Analytical and numerical gradient comparisons: 10 weights for each of the two widths |
| `loss_curves.png` | Training-loss curves for all six configurations |
| `config_resolved.yaml` | Settings, versions, scaling and split evidence, initialization seeds, checks, and raw timings |

## Interpreting the results

Learning rates are compared at the same fixed epoch budget. A lower final training loss need not imply better test accuracy, and the six results describe one data split and one initialization per width. They do not establish a universal best configuration. Numerical gradient checks test selected weights; the derivation and focused tests provide additional evidence about the implementation.

Forward and backward timings depend on matrix sizes, the NumPy backend, and this host. Small operations can be dominated by overhead. A wider network or a backward pass is not guaranteed to be slower in every measurement; conclusions should follow the recorded medians and timing scope.
