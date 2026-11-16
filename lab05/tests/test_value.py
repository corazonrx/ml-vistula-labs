"""Derivative checks for scalar operations, shared graphs, and deep graphs."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lab05.src.value import Value


def assert_topological_order(root: Value) -> list[Value]:
    order = root.topological_order()
    positions = {id(node): position for position, node in enumerate(order)}
    assert len(order) == len(positions), "Shared nodes must appear exactly once."
    assert order[-1] is root
    for node in order:
        assert isinstance(node.parents, tuple)
        for parent in node.parents:
            assert positions[id(parent)] < positions[id(node)]
    return order


def test_shared_subgraphs_and_repeated_parents_accumulate_each_edge():
    x = Value(2.0, label="x")
    y = Value(3.0, label="y")
    shared = x * y
    output = shared + shared + x * x

    output.backward()

    assert output.data == pytest.approx(16.0)
    assert output.grad == pytest.approx(1.0)
    assert shared.grad == pytest.approx(2.0)
    assert x.grad == pytest.approx(10.0)
    assert y.grad == pytest.approx(4.0)
    assert_topological_order(output)


def test_backward_is_fresh_per_call_and_zero_grad_only_clears_its_value():
    x = Value(2.0)
    shared = x * x
    output = shared + shared
    output.backward()
    first_gradients = [node.grad for node in output.topological_order()]

    output.backward()

    assert [node.grad for node in output.topological_order()] == pytest.approx(first_gradients)
    assert x.grad == pytest.approx(8.0)
    x.zero_grad()
    assert x.grad == 0.0
    assert shared.grad == pytest.approx(2.0)
    output.backward()
    assert x.grad == pytest.approx(8.0)


def test_leaf_backward_and_sum_identity():
    leaf = Value(4.0)
    assert leaf.sum() is leaf
    leaf.backward()
    assert leaf.grad == 1.0
    assert_topological_order(leaf)


@pytest.mark.parametrize(
    "data, exponent, expected_value, expected_gradient",
    [(4.0, 0.5, 2.0, 0.25), (2.0, -2.0, 0.25, -0.25),
     (-2.0, 3.0, -8.0, 12.0), (0.0, 0.0, 1.0, 0.0)],
)
def test_real_constant_powers(data, exponent, expected_value, expected_gradient):
    x = Value(data)
    output = x ** exponent
    output.backward()
    assert output.data == pytest.approx(expected_value)
    assert x.grad == pytest.approx(expected_gradient)


@pytest.mark.parametrize("data, expected_gradient", [(-2.0, 0.0), (0.0, 0.0), (2.0, 1.0)])
def test_relu_uses_zero_derivative_at_zero(data, expected_gradient):
    x = Value(data)
    output = x.relu()
    output.backward()
    assert output.data == max(data, 0.0)
    assert x.grad == expected_gradient


def test_reverse_operators_and_negation_obey_the_chain_rule():
    x = Value(2.0)
    output = 8 / x + (5 - x) + 2 * x - x / 2 + -x + x ** 3
    output.backward()
    assert output.data == pytest.approx(16.0)
    assert x.grad == pytest.approx(9.5)


def test_exp_and_log_are_graph_operations():
    x = Value(0.7)
    exponential = x.exp()
    output = exponential.log()
    output.backward()
    assert exponential.data == pytest.approx(math.exp(0.7))
    assert output.data == pytest.approx(0.7)
    assert x.grad == pytest.approx(1.0)
    assert_topological_order(output)


@pytest.mark.parametrize("data", [-1000.0, -2.0, 0.0, 2.0, 1000.0])
def test_sigmoid_is_stable_and_differentiable(data):
    x = Value(data)
    output = x.sigmoid()
    output.backward()
    expected = 1.0 / (1.0 + math.exp(-data)) if data >= 0 else math.exp(data) / (1.0 + math.exp(data))
    assert math.isfinite(output.data)
    assert 0.0 <= output.data <= 1.0
    assert output.data == pytest.approx(expected)
    assert x.grad == pytest.approx(expected * (1.0 - expected))


def test_small_vector_expression_matches_central_differences_and_pytorch_float64():
    a_values = [1.2, -0.7, 0.8]
    b_values = [-0.5, 1.3, 0.4]
    c_values = [0.2, -0.1, 0.9]
    a = [Value(value) for value in a_values]
    b = [Value(value) for value in b_values]
    c = [Value(value) for value in c_values]
    output = sum(((ai * bi + ci) ** 2).relu() for ai, bi, ci in zip(a, b, c))
    output.backward()

    torch_a = torch.tensor(a_values, dtype=torch.float64, requires_grad=True)
    torch_b = torch.tensor(b_values, dtype=torch.float64, requires_grad=True)
    torch_c = torch.tensor(c_values, dtype=torch.float64, requires_grad=True)
    torch_output = ((torch_a * torch_b + torch_c) ** 2).relu().sum()
    torch_output.backward()

    def evaluate(flat: list[float]) -> float:
        av, bv, cv = flat[:3], flat[3:6], flat[6:]
        return sum(max((ai * bi + ci) ** 2, 0.0) for ai, bi, ci in zip(av, bv, cv))

    flat = a_values + b_values + c_values
    epsilon = 1e-6
    numerical_gradients = []
    for index in range(len(flat)):
        plus, minus = flat.copy(), flat.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical_gradients.append((evaluate(plus) - evaluate(minus)) / (2.0 * epsilon))
    engine_gradients = [node.grad for node in a + b + c]
    torch_gradients = torch_a.grad.tolist() + torch_b.grad.tolist() + torch_c.grad.tolist()

    assert output.data == pytest.approx(torch_output.item(), abs=1e-12)
    assert engine_gradients == pytest.approx(torch_gradients, rel=1e-12, abs=1e-12)
    assert engine_gradients == pytest.approx(numerical_gradients, rel=1e-6, abs=1e-8)


def test_deep_graph_traversal_and_backward_do_not_depend_on_python_recursion():
    leaf = Value(0.0)
    output = leaf
    for _ in range(3000):
        output = output + 1.0
    order = assert_topological_order(output)
    output.backward()
    assert len(order) >= 3001
    assert output.data == pytest.approx(3000.0)
    assert leaf.grad == pytest.approx(1.0)


@pytest.mark.parametrize("data", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_are_rejected(data):
    with pytest.raises(ValueError):
        Value(data)


@pytest.mark.parametrize("data", [0.0, -1.0])
def test_log_rejects_nonpositive_values(data):
    with pytest.raises(ValueError):
        Value(data).log()


@pytest.mark.parametrize("data, exponent", [(-2.0, 0.5), (0.0, -1.0)])
def test_power_rejects_nonreal_or_undefined_domains(data, exponent):
    with pytest.raises(ValueError):
        Value(data) ** exponent


def test_power_requires_a_real_constant_exponent():
    with pytest.raises(TypeError):
        Value(2.0) ** Value(3.0)
    with pytest.raises(TypeError):
        Value(2.0) ** "3"
