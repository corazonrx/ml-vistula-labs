"""Check dimensions, numerical stability, manual derivatives, and SGD state."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lab02.src.gradient_check import check_gradients
from lab02.src.network import NumpyMLP, cross_entropy


@pytest.mark.parametrize("width, expected_count", [(16, 1482), (64, 8970)])
def test_required_architectures_initialize_shapes_distribution_and_probabilities(width, expected_count):
    model = NumpyMLP(width, seed=42)
    repeated = NumpyMLP(width, seed=42)
    expected = {"W1": (64, width), "b1": (width,), "W2": (width, width),
                "b2": (width,), "W3": (width, 10), "b3": (10,)}
    for name, shape in expected.items():
        assert model.params[name].shape == shape
        assert model.params[name].dtype == np.float64
        np.testing.assert_array_equal(model.params[name], repeated.params[name])
        if name.startswith("b"):
            np.testing.assert_array_equal(model.params[name], 0)
    assert model.params["W1"].std() == pytest.approx(0.01, rel=0.1)
    assert abs(model.params["W1"].mean()) < 0.001
    assert model.parameter_count == expected_count
    inputs = np.random.default_rng(9).uniform(size=(7, 64))
    probabilities, cache = model.forward(inputs)
    assert probabilities.shape == (7, 10)
    assert set(cache) == {"X", "Z1", "A1", "Z2", "A2", "Z3", "P"}
    assert probabilities is cache["P"]
    assert np.isfinite(probabilities).all()
    assert (probabilities > 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6)
    targets = np.eye(10)[np.arange(7)]
    assert cross_entropy(cache["Z3"], targets) == pytest.approx(math.log(10), abs=1e-3)


def test_softmax_and_logit_cross_entropy_are_stable_for_large_common_offsets():
    model = NumpyMLP(3)
    for parameter in model.params.values():
        parameter.fill(0)
    model.params["b3"][...] = 1e12 + np.arange(10)
    probabilities, cache = model.forward(np.ones((2, 64)))
    expected = np.exp(np.arange(10) - 9)
    expected /= expected.sum()
    np.testing.assert_allclose(probabilities, np.tile(expected, (2, 1)), rtol=0, atol=1e-15)
    targets = np.eye(10)[[0, 9]]
    expected_loss = -np.log(expected[[0, 9]]).mean()
    assert cross_entropy(cache["Z3"], targets) == pytest.approx(expected_loss, abs=1e-14)
    assert cross_entropy(np.full((2, 10), 1e300), targets) == pytest.approx(math.log(10), abs=1e-14)
    assert cross_entropy(np.zeros((2, 10)), targets) == pytest.approx(math.log(10), abs=1e-14)
    extreme = np.full((1, 10), -10000.0)
    extreme[0, 0] = 10000.0
    assert cross_entropy(extreme, np.eye(10)[[1]]) == pytest.approx(20000.0)


def test_chain_rule_matches_central_differences_for_every_parameter_type():
    rng = np.random.default_rng(14)
    model = NumpyMLP(3)
    model.params["W1"][...] = rng.uniform(0.01, 0.05, size=(64, 3))
    model.params["W2"][...] = rng.uniform(0.01, 0.05, size=(3, 3))
    model.params["W3"][...] = rng.normal(0, 0.1, size=(3, 10))
    model.params["b1"].fill(0.2)
    model.params["b2"].fill(0.2)
    inputs = rng.uniform(0.1, 0.9, size=(4, 64))
    targets = np.eye(10)[[0, 1, 3, 7]]
    original_inputs, original_targets = inputs.copy(), targets.copy()
    originals = {name: array.copy() for name, array in model.params.items()}
    probabilities, cache = model.forward(inputs)
    cache_before = {name: array.copy() for name, array in cache.items()}
    gradients = model.backward(cache, targets)
    assert set(gradients) == set(model.params)
    np.testing.assert_allclose(gradients["b3"], (probabilities - targets).mean(axis=0), atol=1e-15)
    for name, parameter in model.params.items():
        assert gradients[name].shape == parameter.shape
        assert gradients[name].dtype == np.float64
        assert np.isfinite(gradients[name]).all()
        for index in [tuple(0 for _ in parameter.shape), tuple(size - 1 for size in parameter.shape)]:
            original = parameter[index]
            epsilon = 1e-5
            try:
                parameter[index] = original + epsilon
                plus = cross_entropy(model.forward(inputs)[1]["Z3"], targets)
                parameter[index] = original - epsilon
                minus = cross_entropy(model.forward(inputs)[1]["Z3"], targets)
            finally:
                parameter[index] = original
            numeric = (plus - minus) / (2 * epsilon)
            assert gradients[name][index] == pytest.approx(numeric, rel=2e-7, abs=1e-10)
        np.testing.assert_array_equal(parameter, originals[name])
    np.testing.assert_array_equal(inputs, original_inputs)
    np.testing.assert_array_equal(targets, original_targets)
    for name, value in cache.items():
        np.testing.assert_array_equal(value, cache_before[name])


def test_sgd_updates_in_place_and_checks_all_shapes_before_mutating_parameters():
    model = NumpyMLP(3)
    originals = {name: parameter.copy() for name, parameter in model.params.items()}
    identities = {name: id(parameter) for name, parameter in model.params.items()}
    gradients = {name: np.full_like(parameter, 0.25) for name, parameter in model.params.items()}
    malformed = dict(gradients, W3=np.zeros((1, 1)))
    with pytest.raises(ValueError, match="W3"):
        model.step(malformed, 0.1)
    for name, parameter in model.params.items():
        np.testing.assert_array_equal(parameter, originals[name])
    model.step(gradients, 0.1)
    for name, parameter in model.params.items():
        assert id(parameter) == identities[name]
        np.testing.assert_allclose(parameter, originals[name] - 0.025, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(gradients[name], 0.25)


@pytest.mark.parametrize("width", [16, 64])
def test_unfiltered_weight_check_passes_required_digits_experiment_and_restores_parameters(width):
    inputs, labels = load_digits(return_X_y=True)
    train_x, _, train_y, _ = train_test_split(
        inputs / 16.0, labels, test_size=0.3, stratify=labels, random_state=42
    )
    model = NumpyMLP(width, seed=42)
    before = {name: parameter.copy() for name, parameter in model.params.items()}
    rows, detail = check_gradients(model, train_x[:16], np.eye(10)[train_y[:16]])
    assert len(rows) == 10
    assert detail["passed"] and detail["passed_weights"] >= 9
    assert detail["required_passes"] == 9
    total = sum(model.params[name].size for name in ("W1", "W2", "W3"))
    assert detail["selected_flat_indices"] == np.random.default_rng(42).choice(total, 10, replace=False).tolist()
    assert detail["analytic_zero_count"] == sum(row["analytic_grad"] == 0 for row in rows)
    assert detail["numerical_zero_count"] == sum(row["numerical_grad"] == 0 for row in rows)
    for row in rows:
        expected_error = abs(row["analytic_grad"] - row["numerical_grad"]) / max(
            1e-12, abs(row["analytic_grad"]) + abs(row["numerical_grad"])
        )
        assert row["relative_error"] == expected_error
        assert row["passed"] == (expected_error < 1e-6)
    assert detail["parameters_restored"]
    for name, parameter in model.params.items():
        np.testing.assert_array_equal(parameter, before[name])


def test_gradient_check_restores_every_parameter_when_an_evaluation_raises(monkeypatch):
    model = NumpyMLP(3)
    originals = {name: parameter.copy() for name, parameter in model.params.items()}

    def interrupted_loss(*args):
        raise RuntimeError("interrupted numerical evaluation")

    monkeypatch.setattr("lab02.src.gradient_check.cross_entropy", interrupted_loss)
    with pytest.raises(RuntimeError, match="interrupted"):
        check_gradients(model, np.ones((2, 64)), np.eye(10)[[0, 1]])
    for name, parameter in model.params.items():
        np.testing.assert_array_equal(parameter, originals[name])
