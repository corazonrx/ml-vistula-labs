"""A small scalar reverse-mode automatic differentiation engine.

Each operation creates a ``Value`` node with ordered input nodes and a local
chain-rule function. ``backward()`` resets gradients on the whole reachable graph
before differentiating, so repeated calls compute a fresh derivative rather than
accumulating results from previous calls. Repeated edges within one graph still
accumulate correctly, including ``x * x`` and shared intermediate expressions.

Backward closures accept an upstream scalar gradient. They capture parents and
cached forward scalars, never their output node; the graph therefore has no
output/closure reference cycles requiring cyclic garbage collection.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Callable, Iterable


def _real(value: Real, description: str) -> float:
    """Convert supported real constants while rejecting nonfinite values."""
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{description} must be a real numeric scalar, got {type(value).__name__}")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{description} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite, got {result}")
    return result


def _accumulate(parent: Value, contribution: float) -> None:
    updated = parent.grad + contribution
    if not math.isfinite(updated):
        raise ValueError("backward produced a nonfinite gradient")
    parent.grad = updated


class Value:
    """One finite real scalar, its gradient, and its local computation graph."""

    __slots__ = ("data", "grad", "parents", "_backward", "_op", "label")

    def __init__(
        self,
        data: Real,
        _children: Iterable[Value] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data = _real(data, "Value data")
        self.grad = 0.0
        self.parents = tuple(_children)
        if not all(isinstance(parent, Value) for parent in self.parents):
            raise TypeError("Value parents must be Value nodes")
        self._op = _op
        self.label = label
        self._backward: Callable[[float], None] = lambda upstream: None

    @staticmethod
    def _coerce(other: Value | Real) -> Value:
        return other if isinstance(other, Value) else Value(other)

    def __repr__(self) -> str:
        label = f", label={self.label!r}" if self.label else ""
        return f"Value(data={self.data!r}, grad={self.grad!r}{label})"

    def __add__(self, other: Value | Real) -> Value:
        other = self._coerce(other)
        result = Value(self.data + other.data, (self, other), "+")

        def backward(upstream: float) -> None:
            _accumulate(self, upstream)
            _accumulate(other, upstream)

        result._backward = backward
        return result

    def __radd__(self, other: Value | Real) -> Value:
        return self._coerce(other) + self

    def __mul__(self, other: Value | Real) -> Value:
        other = self._coerce(other)
        left, right = self.data, other.data
        result = Value(left * right, (self, other), "*")

        def backward(upstream: float) -> None:
            _accumulate(self, right * upstream)
            _accumulate(other, left * upstream)

        result._backward = backward
        return result

    def __rmul__(self, other: Value | Real) -> Value:
        return self._coerce(other) * self

    def __pow__(self, exponent: Real) -> Value:
        """Raise to a finite real constant; the exponent is not a graph node.

        Negative bases require integer exponents. ``0 ** 0`` follows Python and
        PyTorch's value of 1 with zero derivative. A zero base with exponent in
        (0, 1) is rejected because its derivative is not finite.
        """
        power = _real(exponent, "power exponent")
        base = self.data
        if base < 0.0 and not power.is_integer():
            raise ValueError("a negative base requires an integer power exponent")
        if base == 0.0 and power < 0.0:
            raise ValueError("zero cannot be raised to a negative power")
        if base == 0.0 and 0.0 < power < 1.0:
            raise ValueError("zero raised to a power between 0 and 1 has a nonfinite derivative")
        try:
            output = math.pow(base, power)
            derivative = 0.0 if power == 0.0 else power * math.pow(base, power - 1.0)
        except (OverflowError, ValueError) as error:
            raise ValueError("power result and derivative must be finite real numbers") from error
        if not math.isfinite(derivative):
            raise ValueError("power derivative must be finite")
        result = Value(output, (self,), f"**{power:g}")

        def backward(upstream: float) -> None:
            _accumulate(self, derivative * upstream)

        result._backward = backward
        return result

    def __neg__(self) -> Value:
        result = Value(-self.data, (self,), "neg")

        def backward(upstream: float) -> None:
            _accumulate(self, -upstream)

        result._backward = backward
        return result

    def __sub__(self, other: Value | Real) -> Value:
        return self + (-self._coerce(other))

    def __rsub__(self, other: Value | Real) -> Value:
        return self._coerce(other) - self

    def __truediv__(self, other: Value | Real) -> Value:
        return self * self._coerce(other) ** -1

    def __rtruediv__(self, other: Value | Real) -> Value:
        return self._coerce(other) / self

    def relu(self) -> Value:
        """Return max(0, x), choosing derivative zero at x == 0."""
        positive = self.data > 0.0
        result = Value(self.data if positive else 0.0, (self,), "ReLU")

        def backward(upstream: float) -> None:
            _accumulate(self, upstream if positive else 0.0)

        result._backward = backward
        return result

    def exp(self) -> Value:
        try:
            output = math.exp(self.data)
        except OverflowError as error:
            raise ValueError("exp result must be finite") from error
        result = Value(output, (self,), "exp")

        def backward(upstream: float) -> None:
            _accumulate(self, output * upstream)

        result._backward = backward
        return result

    def log(self) -> Value:
        if self.data <= 0.0:
            raise ValueError("log requires a strictly positive input")
        derivative = 1.0 / self.data
        if not math.isfinite(derivative):
            raise ValueError("log derivative must be finite")
        result = Value(math.log(self.data), (self,), "log")

        def backward(upstream: float) -> None:
            _accumulate(self, derivative * upstream)

        result._backward = backward
        return result

    def sigmoid(self) -> Value:
        """Compute the logistic function without exponentiating a large input."""
        if self.data >= 0.0:
            exponential = math.exp(-self.data)
            output = 1.0 / (1.0 + exponential)
        else:
            exponential = math.exp(self.data)
            output = exponential / (1.0 + exponential)
        derivative = output * (1.0 - output)
        result = Value(output, (self,), "sigmoid")

        def backward(upstream: float) -> None:
            _accumulate(self, derivative * upstream)

        result._backward = backward
        return result

    def sum(self) -> Value:
        """Scalar sum is the identity; use built-in sum for a sequence of Values."""
        return self

    def topological_order(self) -> list[Value]:
        """Return unique reachable nodes with every parent before its child.

        Iterative depth-first traversal avoids Python's recursion-depth limit.
        The public ``parents`` tuple retains repeated edges; the traversal visits
        each node once and each local backward rule accounts for its own edges.
        """
        order: list[Value] = []
        visited: set[Value] = set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif node not in visited:
                visited.add(node)
                stack.append((node, True))
                for parent in reversed(node.parents):
                    if parent not in visited:
                        stack.append((parent, False))
        return order

    def zero_grad(self) -> None:
        """Reset this node's gradient; does not traverse its graph."""
        self.grad = 0.0

    def backward(self) -> None:
        """Differentiate this scalar with seed 1, freshly resetting its graph."""
        order = self.topological_order()
        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            node._backward(node.grad)
