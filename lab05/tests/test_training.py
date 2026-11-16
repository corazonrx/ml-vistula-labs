"""Verify matched SGD behavior and that scaling excludes held-out records."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.datasets import make_moons

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lab05.src.training import (
    ScalarMLP,
    initial_parameters,
    run_training,
    scalar_bce,
    train_scalar,
    train_torch,
)


def _scalar_parameters(model):
    return np.asarray([parameter.data for parameter in model.parameters()])


def _torch_parameters(model):
    return np.concatenate([parameter.detach().numpy().ravel() for parameter in model.parameters()])


def test_fresh_scalar_and_torch_models_match_initial_loss_and_three_sgd_steps():
    initial = initial_parameters(seed=17)
    untouched = {name: values.copy() for name, values in initial.items()}
    inputs = np.asarray([
        [-1.2, 0.3], [-0.7, -1.1], [0.2, -0.8], [1.4, -0.2],
        [0.8, 1.3], [-0.4, 0.9], [1.2, 0.7], [-1.3, -0.5],
    ], dtype=np.float64)
    targets = np.asarray([0, 0, 0, 1, 1, 0, 1, 0])
    original_inputs = inputs.copy()
    expected_initial_loss = scalar_bce(ScalarMLP(initial), inputs, targets).data

    scalar, scalar_run = train_scalar(initial, inputs, targets, epochs=3, learning_rate=0.15)
    native, native_run = train_torch(initial, inputs, targets, epochs=3, learning_rate=0.15)

    assert len(scalar.parameters()) == 33
    assert _torch_parameters(native).shape == (33,)
    assert scalar_run["initial_loss"] == pytest.approx(expected_initial_loss, abs=1e-12)
    assert native_run["initial_loss"] == pytest.approx(expected_initial_loss, abs=1e-12)
    assert scalar_run["final_loss"] < scalar_run["initial_loss"]
    assert native_run["final_loss"] < native_run["initial_loss"]
    assert scalar_run["final_loss"] == pytest.approx(native_run["final_loss"], abs=1e-12)
    np.testing.assert_allclose(_scalar_parameters(scalar), _torch_parameters(native), rtol=0, atol=1e-12)
    assert len(scalar_run["loss_history"]) == len(native_run["loss_history"]) == 3
    np.testing.assert_allclose(scalar_run["loss_history"], native_run["loss_history"], rtol=0, atol=1e-12)
    with torch.no_grad():
        native_probabilities = native(torch.as_tensor(inputs)).numpy().ravel()
    np.testing.assert_allclose(scalar.probabilities(inputs), native_probabilities, rtol=0, atol=1e-12)
    for name in initial:
        np.testing.assert_array_equal(initial[name], untouched[name])
    np.testing.assert_array_equal(inputs, original_inputs)


def test_one_step_updates_every_scalar_parameter_when_all_relu_units_are_active():
    # Every hidden activation is 1.25, giving logit=1.0. All 33 analytical
    # gradients are positive for this class-zero sample, so none can be masked
    # by an inactive ReLU or hidden by a zero input/weight.
    initial = {
        "w1": np.full((8, 2), 0.25, dtype=np.float64),
        "b1": np.full(8, 0.5, dtype=np.float64),
        "w2": np.full((1, 8), 0.1, dtype=np.float64),
        "b2": np.zeros(1, dtype=np.float64),
    }
    inputs, targets = [[1.0, 2.0]], [0]
    untrained = ScalarMLP(initial)
    objective = scalar_bce(untrained, inputs, targets)
    objective.backward()
    gradients = np.asarray([parameter.grad for parameter in untrained.parameters()])
    probability = 1 / (1 + np.exp(-1.0))
    expected_gradients = np.concatenate([
        np.tile(probability * 0.1 * np.asarray([1.0, 2.0]), 8),
        np.full(8, probability * 0.1),
        np.full(8, probability * 1.25),
        [probability],
    ])
    np.testing.assert_allclose(gradients, expected_gradients, rtol=0, atol=1e-12)
    assert np.all(gradients > 0)
    initial_weights = _scalar_parameters(untrained)
    trained, run = train_scalar(initial, inputs, targets, epochs=1, learning_rate=0.05)
    np.testing.assert_allclose(
        _scalar_parameters(trained), initial_weights - 0.05 * expected_gradients,
        rtol=0, atol=1e-12,
    )
    assert np.all(_scalar_parameters(trained) < initial_weights)
    assert run["final_loss"] < run["initial_loss"]


def test_training_pipeline_keeps_source_ids_separate_and_fits_scaler_on_train_only():
    config = {
        "data": {"samples": 20, "noise": 0.2, "test_size": 0.3, "random_state": 42},
        "model": {"hidden": 8},
        "training": {
            "initialization_seed": 42, "warmup_epochs": 1,
            "epochs": 3, "learning_rate": 0.15, "repeats": 1,
        },
    }
    rows, detail, native = run_training(config)
    original_inputs, original_targets = make_moons(n_samples=20, noise=0.2, random_state=42)
    data = detail["data"]
    train_ids = data["train_source_ids"]
    test_ids = data["test_source_ids"]
    assert set(train_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(test_ids) == set(range(20))
    assert len(train_ids) == data["train_rows"] == data["scaler_fit_rows"] == 14
    assert len(test_ids) == data["test_rows"] == 6
    assert data["scaler_fit_partition"] == "train only"
    np.testing.assert_allclose(data["scaler_mean"], original_inputs[train_ids].mean(axis=0))
    np.testing.assert_allclose(data["scaler_scale"], original_inputs[train_ids].std(axis=0))
    assert not np.allclose(data["scaler_mean"], original_inputs.mean(axis=0))
    assert data["train_class_counts"] == np.bincount(original_targets[train_ids]).tolist()
    assert data["test_class_counts"] == np.bincount(original_targets[test_ids]).tolist()
    assert {row["framework"] for row in rows} == {"scalar_autodiff", "pytorch"}
    assert all(row["parameters"] == 33 and row["final_bce"] < row["initial_bce"] for row in rows)
    assert detail["max_parameter_difference"] < 1e-10
    assert detail["max_test_probability_difference"] < 1e-10
    assert detail["max_training_loss_difference"] < 1e-10
    assert native.training is False
    assert all(parameter.dtype == torch.float64 and parameter.device.type == "cpu" for parameter in native.parameters())
