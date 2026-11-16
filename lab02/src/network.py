"""A two-hidden-layer classifier using explicit NumPy chain-rule gradients."""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np


PARAMETER_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")
WEIGHT_NAMES = ("W1", "W2", "W3")


def _matrix(values, columns: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != columns or array.shape[0] == 0:
        raise ValueError(f"{name} must have nonempty shape (batch, {columns})")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _targets(values, rows: int) -> np.ndarray:
    targets = _matrix(values, 10, "one_hot")
    if targets.shape[0] != rows:
        raise ValueError("one_hot batch size must match the input batch")
    if np.any(targets < 0) or not np.allclose(targets.sum(axis=1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("one_hot rows must be nonnegative and sum to one")
    return targets


def cross_entropy(logits, one_hot) -> float:
    """Mean categorical cross-entropy, calculated from logits without clipping.

    Subtracting each row's maximum before log-sum-exp prevents exponential
    overflow and preserves the loss even when logits share a large offset.
    """
    scores = _matrix(logits, 10, "logits")
    targets = _targets(one_hot, scores.shape[0])
    shifted = scores - scores.max(axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=1))
    loss = float(np.mean(log_normalizer - np.sum(targets * shifted, axis=1)))
    if not math.isfinite(loss):
        raise FloatingPointError("cross-entropy produced a nonfinite loss")
    return loss


class NumpyMLP:
    """64 -> width ReLU -> width ReLU -> 10 softmax, all float64.

    Weights are independent normal draws with mean zero and the explicit
    standard deviation ``weight_std`` (default 0.01); all biases start at zero.
    Parameter arrays remain the same objects during in-place SGD updates.
    """

    cross_entropy = staticmethod(cross_entropy)

    def __init__(self, width: int, seed: int = 42, weight_std: float = 0.01):
        if isinstance(width, bool) or not isinstance(width, Integral) or width < 1:
            raise ValueError("width must be a positive integer")
        if not isinstance(weight_std, Real) or not math.isfinite(weight_std) or weight_std <= 0:
            raise ValueError("weight_std must be finite and positive")
        self.width = int(width)
        self.seed = seed
        self.weight_std = float(weight_std)
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.normal(0.0, self.weight_std, size=(64, self.width)),
            "b1": np.zeros(self.width, dtype=np.float64),
            "W2": rng.normal(0.0, self.weight_std, size=(self.width, self.width)),
            "b2": np.zeros(self.width, dtype=np.float64),
            "W3": rng.normal(0.0, self.weight_std, size=(self.width, 10)),
            "b3": np.zeros(10, dtype=np.float64),
        }

    @property
    def parameter_count(self) -> int:
        return sum(parameter.size for parameter in self.params.values())

    def forward(self, X) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        inputs = _matrix(X, 64, "X")
        params = self.params
        Z1 = inputs @ params["W1"] + params["b1"]
        A1 = np.maximum(Z1, 0.0)
        Z2 = A1 @ params["W2"] + params["b2"]
        A2 = np.maximum(Z2, 0.0)
        Z3 = A2 @ params["W3"] + params["b3"]
        shifted = Z3 - Z3.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("forward pass produced nonfinite probabilities")
        cache = {"X": inputs, "Z1": Z1, "A1": A1, "Z2": Z2,
                 "A2": A2, "Z3": Z3, "P": probabilities}
        return probabilities, cache

    def backward(self, cache: dict[str, np.ndarray], one_hot) -> dict[str, np.ndarray]:
        """Differentiate mean softmax cross-entropy by the chain rule.

        ReLU's derivative is 1 for positive preactivation and 0 otherwise,
        including zero. No forward inputs, cache entries, or parameters change.
        """
        targets = _targets(one_hot, len(cache["X"]))
        params = self.params
        dZ3 = (cache["P"] - targets) / targets.shape[0]
        dW3 = cache["A2"].T @ dZ3
        db3 = dZ3.sum(axis=0)
        dA2 = dZ3 @ params["W3"].T
        dZ2 = dA2 * (cache["Z2"] > 0.0)
        dW2 = cache["A1"].T @ dZ2
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ params["W2"].T
        dZ1 = dA1 * (cache["Z1"] > 0.0)
        dW1 = cache["X"].T @ dZ1
        db1 = dZ1.sum(axis=0)
        gradients = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                     "W3": dW3, "b3": db3}
        if not all(np.isfinite(gradient).all() for gradient in gradients.values()):
            raise FloatingPointError("backward pass produced nonfinite gradients")
        return gradients

    def step(self, grads: dict[str, np.ndarray], lr: float) -> None:
        """Apply in-place SGD after checking every gradient, before any update."""
        if not isinstance(lr, Real) or not math.isfinite(lr) or lr < 0:
            raise ValueError("learning rate must be finite and nonnegative")
        if set(grads) != set(self.params):
            raise ValueError(f"gradients must have exactly these keys: {PARAMETER_NAMES}")
        checked = {}
        for name, parameter in self.params.items():
            gradient = np.asarray(grads[name], dtype=np.float64)
            if gradient.shape != parameter.shape:
                raise ValueError(f"gradient {name} must have shape {parameter.shape}")
            if not np.isfinite(gradient).all():
                raise ValueError(f"gradient {name} must be finite")
            checked[name] = gradient
        for name, parameter in self.params.items():
            parameter[...] -= float(lr) * checked[name]
