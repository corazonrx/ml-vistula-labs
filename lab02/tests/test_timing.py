"""Check independent timing calls, fixed-cache use and numerical invariance."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lab02.src.timing import measure_passes
from lab02.src.network import NumpyMLP


class CountingModel:
    """A deterministic 64 -> 3 -> 3 -> 10 pass with traceable call identities."""

    def __init__(self):
        self.params = {
            "W1": np.full((64, 3), 0.01), "b1": np.zeros(3),
            "W2": np.full((3, 3), 0.02), "b2": np.zeros(3),
            "W3": np.full((3, 10), 0.03), "b3": np.zeros(10),
        }
        self.forward_calls = 0
        self.backward_cache_ids = []

    def forward(self, inputs):
        self.forward_calls += 1
        z1 = inputs @ self.params["W1"] + self.params["b1"]
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.params["W2"] + self.params["b2"]
        a2 = np.maximum(z2, 0)
        z3 = a2 @ self.params["W3"] + self.params["b3"]
        exponentials = np.exp(z3 - z3.max(axis=1, keepdims=True))
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        cache = dict(X=inputs, Z1=z1, A1=a1, Z2=z2, A2=a2, Z3=z3, P=probabilities)
        return probabilities, cache

    def backward(self, cache, labels):
        self.backward_cache_ids.append(id(cache))
        return {name: np.full_like(value, cache["P"].sum() - labels.sum() + 0.25) for name, value in self.params.items()}


def test_each_pass_is_timed_independently_and_backward_reuses_one_cache(monkeypatch):
    model = CountingModel()
    inputs = np.ones((4, 64), dtype=np.float64)
    labels = np.eye(10)[[0, 1, 2, 3]]
    parameters = {name: value.copy() for name, value in model.params.items()}
    # Synthetic clock durations make median/units verifiable without asserting
    # a performance relationship on the executing host.
    clock = iter([0, 1_000_000, 2_000_000, 5_000_000, 6_000_000, 8_000_000,
                  9_000_000, 13_000_000, 14_000_000, 20_000_000, 21_000_000, 26_000_000])
    monkeypatch.setattr("lab02.src.timing.time.perf_counter_ns", lambda: next(clock))
    result = measure_passes(model, inputs, labels, repeats=3, warmup=2)

    assert model.forward_calls == 1 + 2 + 3
    assert len(model.backward_cache_ids) == 1 + 2 + 3
    assert len(set(model.backward_cache_ids)) == 1
    assert result["forward_ms"] == [1.0, 3.0, 2.0]
    assert result["backward_ms"] == [4.0, 6.0, 5.0]
    assert result["median_forward_ms"] == 2.0
    assert result["median_backward_ms"] == 5.0
    assert result["forward_over_backward_ratio"] == 0.4
    assert result["batch_shape"] == [4, 64]
    assert result["dtype"] == "float64"
    assert result["matrix_multiplication_flops"]["forward"] == 2 * 4 * (64 * 3 + 3 * 3 + 3 * 10)
    assert result["matrix_multiplication_flops"]["backward"] == 2 * 4 * (64 * 3 + 2 * 3 * 3 + 2 * 3 * 10)
    for name, before in parameters.items():
        np.testing.assert_array_equal(model.params[name], before)
    np.testing.assert_array_equal(inputs, np.ones((4, 64)))
    np.testing.assert_array_equal(labels, np.eye(10)[[0, 1, 2, 3]])
    json.dumps(result, allow_nan=False)


def test_mutating_backward_cannot_silently_be_reported_as_an_inference_timing():
    class MutatingModel(CountingModel):
        def backward(self, cache, labels):
            gradients = super().backward(cache, labels)
            self.params["b3"] += 1
            return gradients

    with pytest.raises(RuntimeError, match="parameters during setup"):
        measure_passes(MutatingModel(), np.ones((2, 64)), np.eye(10)[[0, 1]], repeats=1, warmup=0)


def test_real_network_timing_preserves_parameters_and_fixed_cache_gradients():
    model = NumpyMLP(width=4, seed=31)
    inputs = np.random.default_rng(9).random((4, 64))
    labels = np.eye(10)[[1, 4, 5, 9]]
    probabilities, cache = model.forward(inputs)
    gradients = model.backward(cache, labels)
    parameters = {name: value.copy() for name, value in model.params.items()}

    result = measure_passes(model, inputs, labels, repeats=2, warmup=1)

    assert len(result["forward_ms"]) == len(result["backward_ms"]) == 2
    assert all(np.isfinite(result["forward_ms"] + result["backward_ms"]))
    assert result["correctness"]["weights_unchanged"] is True
    assert result["correctness"]["gradients_unchanged"] is True
    after_probabilities, after_cache = model.forward(inputs)
    after_gradients = model.backward(after_cache, labels)
    np.testing.assert_array_equal(after_probabilities, probabilities)
    for name in parameters:
        np.testing.assert_array_equal(model.params[name], parameters[name])
        np.testing.assert_array_equal(after_gradients[name], gradients[name])
