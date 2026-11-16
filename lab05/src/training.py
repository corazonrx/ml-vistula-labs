"""Matched full-batch SGD experiments for scalar autodiff and PyTorch."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .value import Value


def initial_parameters(seed=42, hidden=8):
    rng = np.random.default_rng(seed)
    return {
        'w1': rng.uniform(-math.sqrt(6 / (2 + hidden)), math.sqrt(6 / (2 + hidden)), (hidden, 2)),
        'b1': np.zeros(hidden, dtype=np.float64),
        'w2': rng.uniform(-math.sqrt(6 / (hidden + 1)), math.sqrt(6 / (hidden + 1)), (1, hidden)),
        'b2': np.zeros(1, dtype=np.float64),
    }


class ScalarMLP:
    """A 2 -> 8 -> 1 network; every trainable scalar uses our own Value."""
    def __init__(self, initial):
        self.w1 = [[Value(value) for value in row] for row in initial['w1']]
        self.b1 = [Value(value) for value in initial['b1']]
        self.w2 = [Value(value) for value in initial['w2'][0]]
        self.b2 = Value(initial['b2'][0])

    def parameters(self):
        return [value for row in self.w1 for value in row] + self.b1 + self.w2 + [self.b2]

    def __call__(self, inputs):
        hidden = [sum((weight * float(x) for weight, x in zip(row, inputs)), bias).relu()
                  for row, bias in zip(self.w1, self.b1)]
        return sum((weight * value for weight, value in zip(self.w2, hidden)), self.b2).sigmoid()

    def probabilities(self, inputs):
        return np.asarray([self(row).data for row in inputs], dtype=np.float64)


def scalar_bce(model, inputs, targets):
    losses = []
    for row, target in zip(inputs, targets):
        probability = model(row)
        # The labels are binary, so each branch is exactly the nonzero BCE term.
        losses.append(-probability.log() if target == 1 else -(1 - probability).log())
    return sum(losses) / len(losses)


def make_torch_model(initial):
    hidden = len(initial['b1'])
    model = torch.nn.Sequential(
        torch.nn.Linear(2, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1), torch.nn.Sigmoid(),
    ).to(device='cpu', dtype=torch.float64)
    with torch.no_grad():
        for parameter, name in zip(model.parameters(), ['w1', 'b1', 'w2', 'b2']):
            parameter.copy_(torch.as_tensor(initial[name], dtype=torch.float64))
    return model


def train_scalar(initial, inputs, targets, epochs, learning_rate):
    model = ScalarMLP(initial)
    parameters = model.parameters()
    initial_loss = scalar_bce(model, inputs, targets).data
    history = []
    start = time.perf_counter_ns()
    for _ in range(epochs):
        for parameter in parameters:
            parameter.zero_grad()
        loss = scalar_bce(model, inputs, targets)
        loss.backward()
        history.append(loss.data)
        for parameter in parameters:
            parameter.data -= learning_rate * parameter.grad
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    final_loss = scalar_bce(model, inputs, targets).data
    if not math.isfinite(final_loss):
        raise FloatingPointError('Scalar training produced a non-finite loss.')
    return model, {'elapsed_ms': elapsed_ms, 'initial_loss': initial_loss,
                   'final_loss': final_loss, 'loss_history': history}


def train_torch(initial, inputs, targets, epochs, learning_rate):
    model = make_torch_model(initial)
    x = torch.as_tensor(inputs, dtype=torch.float64)
    y = torch.as_tensor(targets, dtype=torch.float64).reshape(-1, 1)
    criterion = torch.nn.BCELoss(reduction='mean')
    with torch.no_grad():
        initial_loss = criterion(model(x), y).item()
    history = []
    start = time.perf_counter_ns()
    for _ in range(epochs):
        for parameter in model.parameters():
            parameter.grad = None
        loss = criterion(model(x), y)
        loss.backward()
        history.append(loss.item())
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(parameter.grad, alpha=-learning_rate)
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    with torch.no_grad():
        final_loss = criterion(model(x), y).item()
    if not math.isfinite(final_loss):
        raise FloatingPointError('PyTorch training produced a non-finite loss.')
    return model, {'elapsed_ms': elapsed_ms, 'initial_loss': initial_loss,
                   'final_loss': final_loss, 'loss_history': history}


def run_training(config):
    settings = config['training']
    data_config = config['data']
    inputs, targets = make_moons(n_samples=data_config['samples'], noise=data_config['noise'], random_state=data_config['random_state'])
    source_ids = np.arange(len(targets))
    train_ids, test_ids = train_test_split(source_ids, test_size=data_config['test_size'], stratify=targets, random_state=data_config['random_state'])
    scaler = StandardScaler().fit(inputs[train_ids])
    train_x, test_x = scaler.transform(inputs[train_ids]), scaler.transform(inputs[test_ids])
    train_y, test_y = targets[train_ids], targets[test_ids]
    initial = initial_parameters(settings['initialization_seed'], config['model']['hidden'])
    scalar_inputs = train_x.tolist()
    scalar_targets = train_y.tolist()
    warmup_epochs = settings['warmup_epochs']
    # Warm-up models are discarded; every measured run starts at the same weights.
    train_scalar(initial, scalar_inputs, scalar_targets, warmup_epochs, settings['learning_rate'])
    train_torch(initial, train_x, train_y, warmup_epochs, settings['learning_rate'])
    measurements = {'scalar_autodiff': [], 'pytorch': []}
    maximum_weight_error = maximum_probability_error = maximum_loss_error = 0.0
    for repeat in range(settings['repeats']):
        outputs = {}
        order = ['scalar_autodiff', 'pytorch'] if repeat % 2 == 0 else ['pytorch', 'scalar_autodiff']
        for framework in order:
            if framework == 'scalar_autodiff':
                model, measurement = train_scalar(initial, scalar_inputs, scalar_targets, settings['epochs'], settings['learning_rate'])
                probabilities = model.probabilities(test_x)
                weights = np.asarray([parameter.data for parameter in model.parameters()])
            else:
                model, measurement = train_torch(initial, train_x, train_y, settings['epochs'], settings['learning_rate'])
                with torch.no_grad():
                    probabilities = model(torch.as_tensor(test_x, dtype=torch.float64)).numpy().ravel()
                weights = np.concatenate([parameter.detach().numpy().ravel() for parameter in model.parameters()])
                final_torch_model = model
            prediction = (probabilities >= 0.5).astype(int)
            measurement['accuracy'] = float(accuracy_score(test_y, prediction))
            measurement['f1'] = float(f1_score(test_y, prediction))
            measurements[framework].append(measurement)
            outputs[framework] = {'probabilities': probabilities, 'weights': weights, 'history': measurement['loss_history']}
        scalar, native = outputs['scalar_autodiff'], outputs['pytorch']
        maximum_weight_error = max(maximum_weight_error, float(np.max(np.abs(scalar['weights'] - native['weights']))))
        maximum_probability_error = max(maximum_probability_error, float(np.max(np.abs(scalar['probabilities'] - native['probabilities']))))
        maximum_loss_error = max(maximum_loss_error, float(np.max(np.abs(np.asarray(scalar['history']) - native['history']))))
        np.testing.assert_allclose(scalar['weights'], native['weights'], rtol=0, atol=1e-10)
        np.testing.assert_allclose(scalar['probabilities'], native['probabilities'], rtol=0, atol=1e-10)
        np.testing.assert_allclose(scalar['history'], native['history'], rtol=0, atol=1e-10)
        print(f"Training repeat {repeat + 1}/{settings['repeats']}: scalar {measurements['scalar_autodiff'][-1]['elapsed_ms']:.1f} ms; PyTorch {measurements['pytorch'][-1]['elapsed_ms']:.1f} ms", flush=True)
    rows = []
    for framework, runs in measurements.items():
        final = runs[-1]
        assert final['final_loss'] < final['initial_loss'], f'{framework} failed to reduce BCE.'
        rows.append({'framework': framework, 'median_training_ms': float(np.median([run['elapsed_ms'] for run in runs])),
                     'initial_bce': final['initial_loss'], 'final_bce': final['final_loss'],
                     'test_accuracy': final['accuracy'], 'test_f1': final['f1'],
                     'epochs': settings['epochs'], 'runs': settings['repeats'], 'parameters': 33})
    detail = {
        'data': {'name': 'make_moons', 'rows': len(targets), 'features': 2,
                 'train_rows': len(train_ids), 'test_rows': len(test_ids),
                 'train_source_ids': train_ids.tolist(), 'test_source_ids': test_ids.tolist(),
                 'train_class_counts': np.bincount(train_y).tolist(), 'test_class_counts': np.bincount(test_y).tolist(),
                 'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
                 'scaler_fit_rows': int(scaler.n_samples_seen_), 'scaler_fit_partition': 'train only'},
        'initial_parameters': {key: value.tolist() for key, value in initial.items()},
        'timing_scope': 'training steps only: gradient reset, forward, mean BCE, backward and full-batch SGD update; initialization, conversion, warm-up and test evaluation excluded',
        'run_order': 'alternating scalar-first and PyTorch-first',
        'warmup_epochs_per_framework': warmup_epochs,
        'max_parameter_difference': maximum_weight_error,
        'max_test_probability_difference': maximum_probability_error,
        'max_training_loss_difference': maximum_loss_error,
        'measurements': {},
    }
    for framework, runs in measurements.items():
        detail['measurements'][framework] = {
            'raw_training_ms': [run['elapsed_ms'] for run in runs],
            'initial_bce': [run['initial_loss'] for run in runs],
            'final_bce': [run['final_loss'] for run in runs],
            'test_accuracy': [run['accuracy'] for run in runs],
            'test_f1': [run['f1'] for run in runs],
            'last_run_loss_history': runs[-1]['loss_history'],
        }
    return rows, detail, final_torch_model.eval()
