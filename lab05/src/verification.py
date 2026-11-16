"""Check the requested expression against PyTorch and central differences."""
from __future__ import annotations

import numpy as np
import torch

from .value import Value


def verify_gradients(config):
    rng = np.random.default_rng(config['seed'])
    inputs = rng.uniform(-0.9, 0.9, size=(3, config['elements']))
    names = ['a', 'b', 'c']
    nodes = [[Value(value, label=f'{name}[{index}]') for index, value in enumerate(row)]
             for name, row in zip(names, inputs)]
    output = sum(((a * b + c) ** 2).relu() for a, b, c in zip(*nodes))
    output.backward()
    native = [torch.tensor(row, dtype=torch.float64, requires_grad=True) for row in inputs]
    native_output = ((native[0] * native[1] + native[2]) ** 2).relu().sum()
    native_output.backward()

    def objective(values):
        return float(np.maximum((values[0] * values[1] + values[2]) ** 2, 0).sum())

    epsilon = config['epsilon']
    rows = []
    for name_index, name in enumerate(names):
        for index in range(inputs.shape[1]):
            plus, minus = inputs.copy(), inputs.copy()
            plus[name_index, index] += epsilon
            minus[name_index, index] -= epsilon
            numeric = (objective(plus) - objective(minus)) / (2 * epsilon)
            scalar = nodes[name_index][index].grad
            pytorch = native[name_index].grad[index].item()
            rows.append({'input': name, 'index': index, 'value': float(inputs[name_index, index]),
                         'scalar_grad': scalar, 'pytorch_grad': pytorch, 'numerical_grad': numeric,
                         'abs_scalar_pytorch': abs(scalar - pytorch),
                         'abs_scalar_numerical': abs(scalar - numeric),
                         'abs_pytorch_numerical': abs(pytorch - numeric)})
    errors = {key: max(row[key] for row in rows)
              for key in ['abs_scalar_pytorch', 'abs_scalar_numerical', 'abs_pytorch_numerical']}
    if not all(error < config['tolerance'] for error in errors.values()):
        raise AssertionError(f'Gradient verification failed: {errors}')
    if abs(output.data - native_output.item()) >= config['tolerance']:
        raise AssertionError('Scalar and PyTorch forward values do not match.')
    graph = output.topological_order()
    identifiers = {node: f'n{index}' for index, node in enumerate(graph)}
    lines = ['SCALAR VALUE GRAPH', 'Expression: sum(relu((a[i] * b[i] + c[i]) ** 2))',
             'Topological order: parents before their outputs. Backward traverses this in reverse.',
             'Each backward() resets reachable gradients and seeds the output gradient with 1.', '']
    for node in graph:
        parents = ', '.join(identifiers[parent] for parent in node.parents)
        lines.append(f"{identifiers[node]}: op={node._op or 'leaf'} label={node.label or '-'} data={node.data:.12g} grad={node.grad:.12g} parents=[{parents}]")
    detail = {
        'expression': '((a*b + c)**2).relu().sum()', 'dtype': 'float64',
        'inputs': {name: row.tolist() for name, row in zip(names, inputs)},
        'epsilon': epsilon, 'tolerance': config['tolerance'],
        'numerical_method': 'central difference (f(x+h)-f(x-h))/(2h)',
        'scalar_output': output.data, 'pytorch_output': native_output.item(),
        'scalar_graph_nodes': len(graph), 'compared_input_gradients': len(rows),
        'max_errors': errors, 'passed': True,
    }
    return rows, detail, '\n'.join(lines) + '\n'
