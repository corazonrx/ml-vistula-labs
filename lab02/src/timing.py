"""Measure inference and fixed-cache backpropagation without parameter updates."""

from __future__ import annotations

import statistics
import time
from typing import Any

import numpy as np


_PARAMETERS = ("W1", "b1", "W2", "b2", "W3", "b3")
_CACHE = ("X", "Z1", "A1", "Z2", "A2", "Z3", "P")


def _snapshot(values: dict[str, np.ndarray], names: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not isinstance(values, dict) or any(name not in values for name in names):
        raise ValueError(f"Expected an array dictionary containing {', '.join(names)}.")
    copied = {}
    for name in names:
        value = np.asarray(values[name])
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite numeric values.")
        copied[name] = value.copy()
    return copied


def _unchanged(before: dict[str, np.ndarray], after: dict[str, np.ndarray], description: str) -> None:
    for name, previous in before.items():
        current = np.asarray(after[name])
        if current.dtype != previous.dtype or not np.array_equal(current, previous):
            raise RuntimeError(f"Timing changed {description}: {name}.")


def _validate_output(probabilities: np.ndarray, inputs: np.ndarray, labels: np.ndarray) -> None:
    if not isinstance(probabilities, np.ndarray) or probabilities.ndim != 2:
        raise ValueError("Forward probabilities must be a two-dimensional NumPy array.")
    if probabilities.shape != labels.shape or probabilities.shape[0] != inputs.shape[0]:
        raise ValueError("Probabilities and one-hot labels must have matching batch/class shapes.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Forward probabilities must be finite.")


def measure_passes(
    model: Any,
    X: np.ndarray,
    Y: np.ndarray,
    repeats: int = 100,
    warmup: int = 20,
) -> dict[str, Any]:
    """Return median forward/backward latency and raw independent call timings.

    Forward includes the model's activations, softmax and cache construction.
    Backward repeatedly consumes one precomputed cache and labels, computing
    parameter gradients without another forward pass, cross-entropy evaluation
    or optimizer update. Setup, warmup, wrapper validation and state copies are
    outside the timed intervals; model-internal validation remains included.
    FLOPs count only leading dense matrix products.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer.")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer.")
    inputs, labels = np.asarray(X), np.asarray(Y)
    if inputs.ndim != 2 or labels.ndim != 2 or inputs.shape[0] < 1 or inputs.shape[0] != labels.shape[0]:
        raise ValueError("X and one-hot Y must be nonempty matrices with the same row count.")
    if not np.isfinite(inputs).all() or not np.isfinite(labels).all():
        raise ValueError("Timing inputs and labels must be finite.")
    input_snapshot = inputs.copy()
    label_snapshot = labels.copy()
    parameters = _snapshot(model.params, _PARAMETERS)

    # Prepare once outside timing. Holding this cache fixed makes the backward
    # measurement independent of forward latency and avoids weight updates.
    probabilities, cache = model.forward(inputs)
    _validate_output(probabilities, inputs, labels)
    cache_snapshot = _snapshot(cache, _CACHE)
    reference_gradients = _snapshot(model.backward(cache, labels), _PARAMETERS)
    for name in _PARAMETERS:
        if reference_gradients[name].shape != parameters[name].shape:
            raise ValueError(f"Gradient shape for {name} does not match its parameter.")
    _unchanged(parameters, model.params, "parameters during setup")
    _unchanged(cache_snapshot, cache, "precomputed cache during setup")

    for _ in range(warmup):
        model.forward(inputs)
    forward_ms = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        last_probabilities, last_cache = model.forward(inputs)
        finished = time.perf_counter_ns()
        forward_ms.append(float((finished - started) / 1_000_000))

    for _ in range(warmup):
        model.backward(cache, labels)
    backward_ms = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        last_gradients = model.backward(cache, labels)
        finished = time.perf_counter_ns()
        backward_ms.append(float((finished - started) / 1_000_000))

    # Validation is deliberately after the timed loops. Pure repeated passes
    # must preserve both the supplied data and the caller's parameter state.
    _validate_output(last_probabilities, inputs, labels)
    _snapshot(last_cache, _CACHE)
    _snapshot(last_gradients, _PARAMETERS)
    _unchanged(parameters, model.params, "parameters")
    _unchanged(reference_gradients, last_gradients, "fixed-cache gradient values")
    _unchanged(cache_snapshot, cache, "precomputed cache")
    _unchanged({"X": input_snapshot, "Y": label_snapshot}, {"X": inputs, "Y": labels}, "input data")
    if not np.array_equal(last_probabilities, probabilities):
        raise RuntimeError("Repeated forward passes changed probabilities without a parameter update.")

    batch, input_width = inputs.shape
    hidden1, hidden2 = cache["A1"].shape[1], cache["A2"].shape[1]
    classes = probabilities.shape[1]
    # Forward: XW1, A1W2, A2W3. Backward: dW3, dA2, dW2, dA1,
    # dW1. The unused input gradient dX is deliberately not computed.
    forward_flops = 2 * batch * (input_width * hidden1 + hidden1 * hidden2 + hidden2 * classes)
    backward_flops = 2 * batch * (input_width * hidden1 + 2 * hidden1 * hidden2 + 2 * hidden2 * classes)
    median_forward = float(statistics.median(forward_ms))
    median_backward = float(statistics.median(backward_ms))
    if median_forward <= 0 or median_backward <= 0:
        raise RuntimeError("The clock did not resolve a positive pass duration.")
    dominant = "forward" if median_forward > median_backward else "backward" if median_backward > median_forward else "equal"
    return {
        "batch_shape": list(inputs.shape),
        "label_shape": list(labels.shape),
        "dtype": str(inputs.dtype),
        "warmup": warmup,
        "repeats": repeats,
        "timing_unit": "milliseconds per complete batch call",
        "clock": "time.perf_counter_ns; each call timed independently",
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "median_forward_ms": median_forward,
        "median_backward_ms": median_backward,
        "forward_over_backward_ratio": float(median_forward / median_backward),
        "dominant_measured_pass": dominant,
        "forward_scope": "forward activations, softmax and cache construction",
        "backward_scope": "parameter gradients from one fixed precomputed cache and labels; no forward pass or unused dX",
        "excluded_from_timing": ["setup", "warmup", "wrapper validation", "cross-entropy evaluation", "parameter updates"],
        "model_internal_validation": "Included when performed by forward/backward; the benchmark does not bypass it.",
        "matrix_multiplication_flops": {
            "forward": int(forward_flops),
            "backward": int(backward_flops),
            "forward_matmuls": 3,
            "backward_matmuls": 5,
            "dimensions": {"batch": batch, "inputs": input_width, "hidden1": hidden1, "hidden2": hidden2, "classes": classes},
            "forward_formula": "2 * batch * (inputs*hidden1 + hidden1*hidden2 + hidden2*classes)",
            "backward_formula": "2 * batch * (inputs*hidden1 + 2*hidden1*hidden2 + 2*hidden2*classes)",
            "convention": "One multiply-add counts as 2 FLOPs; activations, softmax, bias additions and reductions are omitted. Counts do not determine the observed timing order.",
        },
        "correctness": {
            "probabilities_finite": True,
            "gradients_finite": True,
            "weights_unchanged": True,
            "gradients_unchanged": True,
            "gradient_check_definition": "Last timed backward output exactly equals the pre-timing fixed-cache reference gradients.",
            "cache_unchanged": True,
            "inputs_and_labels_unchanged": True,
            "probabilities_unchanged": True,
        },
    }
