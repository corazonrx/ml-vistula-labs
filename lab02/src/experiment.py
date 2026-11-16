"""Load and split digits, then train a network with mini-batch SGD."""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from .network import NumpyMLP, cross_entropy


def prepare_data(config):
    raw = load_digits()
    inputs = np.asarray(raw.data, dtype=np.float64) / config['scale_divisor']
    labels = raw.target.astype(np.int64)
    one_hot = np.eye(10, dtype=np.float64)[labels]
    train_ids, test_ids = train_test_split(np.arange(len(labels)), test_size=config['test_size'],
                                         stratify=labels, random_state=config['random_state'])
    digest = hashlib.sha256(inputs.tobytes() + labels.tobytes()).hexdigest()
    detail = {
        'name': 'load_digits', 'rows': len(labels), 'features': 64, 'classes': list(range(10)),
        'image_shape': [8, 8], 'input_range': [float(inputs.min()), float(inputs.max())],
        'scale_divisor': config['scale_divisor'], 'one_hot_columns': 10,
        'train_rows': len(train_ids), 'test_rows': len(test_ids),
        'train_source_ids': train_ids.tolist(), 'test_source_ids': test_ids.tolist(),
        'train_class_counts': np.bincount(labels[train_ids], minlength=10).tolist(),
        'test_class_counts': np.bincount(labels[test_ids], minlength=10).tolist(),
        'scaled_data_sha256': digest,
        'split': 'stratified 70/30 before training; fixed pixel scaling needs no fit',
    }
    return {'train_x': inputs[train_ids], 'test_x': inputs[test_ids],
            'train_y': one_hot[train_ids], 'test_y': one_hot[test_ids],
            'train_labels': labels[train_ids], 'test_labels': labels[test_ids]}, detail


def probability_check(probabilities, tolerance=1e-6):
    if probabilities.ndim != 2 or probabilities.shape[1] != 10:
        raise AssertionError(f'Expected (batch, 10), got {probabilities.shape}.')
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise AssertionError('Softmax probabilities must be finite and in [0, 1].')
    error = float(np.max(np.abs(probabilities.sum(axis=1) - 1)))
    if error > tolerance:
        raise AssertionError(f'Softmax row-sum error {error} exceeds {tolerance}.')
    return error


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, values in model.params.items():
        digest.update(name.encode())
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def train_model(data, width, learning_rate, config):
    model_settings, training = config['model'], config['training']
    model = NumpyMLP(width, seed=model_settings['initialization_seed'], weight_std=model_settings['weight_std'])
    initial_hash = parameter_hash(model)
    probabilities, cache = model.forward(data['train_x'])
    initial_loss = cross_entropy(cache['Z3'], data['train_y'])
    max_probability_error = probability_check(probabilities, config['checks']['probability_sum_tolerance'])
    if abs(initial_loss - np.log(10)) > config['checks']['initial_loss_absolute_tolerance']:
        raise AssertionError('Untrained loss is not sufficiently close to ln(10).')
    rng = np.random.default_rng(training['shuffle_seed'])
    history = []
    steps = 0
    last_batch_size = 0
    for epoch in range(1, training['epochs'] + 1):
        order = rng.permutation(len(data['train_x']))
        for offset in range(0, len(order), training['batch_size']):
            batch = order[offset:offset + training['batch_size']]
            _, batch_cache = model.forward(data['train_x'][batch])
            gradients = model.backward(batch_cache, data['train_y'][batch])
            model.step(gradients, learning_rate)
            steps += 1
            last_batch_size = len(batch)
        train_probabilities, train_cache = model.forward(data['train_x'])
        max_probability_error = max(max_probability_error, probability_check(train_probabilities, config['checks']['probability_sum_tolerance']))
        loss = cross_entropy(train_cache['Z3'], data['train_y'])
        accuracy = float(np.mean(train_probabilities.argmax(axis=1) == data['train_labels']))
        history.append({'width': width, 'learning_rate': learning_rate, 'epoch': epoch,
                        'train_loss': loss, 'train_accuracy': accuracy})
        print(f'W={width:2d} lr={learning_rate:g} epoch={epoch:02d}/{training["epochs"]} loss={loss:.6f} train_accuracy={accuracy:.4f}', flush=True)
    test_probabilities, test_cache = model.forward(data['test_x'])
    max_probability_error = max(max_probability_error, probability_check(test_probabilities, config['checks']['probability_sum_tolerance']))
    test_accuracy = float(np.mean(test_probabilities.argmax(axis=1) == data['test_labels']))
    final_loss = history[-1]['train_loss']
    row = {'width': width, 'learning_rate': learning_rate, 'parameters': model.parameter_count,
           'initial_loss': initial_loss, 'final_train_loss': final_loss,
           'final_train_accuracy': history[-1]['train_accuracy'], 'test_accuracy': test_accuracy,
           'test_loss': cross_entropy(test_cache['Z3'], data['test_y']),
           'epochs': training['epochs'], 'batch_size': training['batch_size']}
    detail = {
        'width': width, 'learning_rate': learning_rate, 'parameter_count': model.parameter_count,
        'parameter_shapes': {name: list(values.shape) for name, values in model.params.items()},
        'initial_parameters_sha256': initial_hash, 'final_parameters_sha256': parameter_hash(model),
        'initial_loss': initial_loss, 'ln10_absolute_difference': abs(initial_loss - float(np.log(10))),
        'max_probability_row_sum_error': max_probability_error,
        'epochs': training['epochs'], 'batches_per_epoch': int(np.ceil(len(data['train_x']) / training['batch_size'])),
        'total_updates': steps, 'last_batch_size': last_batch_size,
        'shuffle_seed': training['shuffle_seed'], 'test_used_for_training_or_epoch_logging': False,
    }
    return model, row, history, detail
