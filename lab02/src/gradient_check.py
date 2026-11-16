"""Unscreened central-difference checks on uniformly sampled model weights."""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np

from .network import NumpyMLP, WEIGHT_NAMES, cross_entropy


def check_gradients(
    model: NumpyMLP,
    X,
    Y,
    seed: int = 42,
    samples: int = 10,
    epsilon: float = 1e-5,
    tolerance: float = 1e-6,
) -> tuple[list[dict], dict]:
    """Check random weight entries without screening or changing the model.

    Candidates are the uniform flattened union of W1, W2 and W3, excluding
    biases. A fixed generator draws without replacement before derivatives are
    inspected. A check passes only when its relative error is strictly below
    tolerance. The overall requirement is at least ceil(0.9 * samples) passes.
    Every parameter is restored in a finally block, including on exceptions.
    """
    total = sum(model.params[name].size for name in WEIGHT_NAMES)
    if isinstance(samples, bool) or not isinstance(samples, Integral) or not 1 <= samples <= total:
        raise ValueError(f"samples must be an integer from 1 through {total}")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    selection = np.random.default_rng(seed).choice(total, size=samples, replace=False)
    offsets = np.cumsum([0] + [model.params[name].size for name in WEIGHT_NAMES])
    originals = {name: parameter.copy() for name, parameter in model.params.items()}
    rows = []
    try:
        _, cache = model.forward(X)
        analytical = model.backward(cache, Y)
        for flat_index in selection:
            array_index = int(np.searchsorted(offsets, flat_index, side="right") - 1)
            name = WEIGHT_NAMES[array_index]
            parameter = model.params[name]
            index = np.unravel_index(int(flat_index - offsets[array_index]), parameter.shape)
            original = float(parameter[index])
            parameter[index] = original + epsilon
            _, plus_cache = model.forward(X)
            plus_loss = cross_entropy(plus_cache["Z3"], Y)
            parameter[index] = original - epsilon
            _, minus_cache = model.forward(X)
            minus_loss = cross_entropy(minus_cache["Z3"], Y)
            parameter[index] = original
            numerical = (plus_loss - minus_loss) / (2.0 * epsilon)
            analytic = float(analytical[name][index])
            absolute = abs(analytic - numerical)
            relative = absolute / max(1e-12, abs(analytic) + abs(numerical))
            rows.append({
                "parameter": name, "row": int(index[0]), "column": int(index[1]),
                "flat_index": int(flat_index), "weight_value": original,
                "analytic_grad": analytic, "numerical_grad": float(numerical),
                "absolute_error": float(absolute), "relative_error": float(relative),
                "passed": bool(relative < tolerance),
                "analytic_is_zero": analytic == 0.0,
                "numerical_is_zero": numerical == 0.0,
            })
    finally:
        for name, parameter in model.params.items():
            parameter[...] = originals[name]
    passed = sum(row["passed"] for row in rows)
    required = math.ceil(0.9 * samples)
    detail = {
        "seed": int(seed), "samples": int(samples), "epsilon": float(epsilon),
        "tolerance": float(tolerance), "strict_comparison": "relative_error < tolerance",
        "relative_error_definition": "abs(analytic-numerical)/max(1e-12,abs(analytic)+abs(numerical))",
        "selection": "uniform without replacement from flattened W1, W2, W3; biases excluded; no screening",
        "candidate_weights": int(total), "selected_flat_indices": selection.tolist(),
        "selected_per_array": {name: sum(row["parameter"] == name for row in rows) for name in WEIGHT_NAMES},
        "passed_weights": int(passed), "required_passes": int(required),
        "passed": passed >= required,
        "max_relative_error": max(row["relative_error"] for row in rows),
        "max_absolute_error": max(row["absolute_error"] for row in rows),
        "analytic_zero_count": sum(row["analytic_is_zero"] for row in rows),
        "numerical_zero_count": sum(row["numerical_is_zero"] for row in rows),
        "both_zero_count": sum(row["analytic_is_zero"] and row["numerical_is_zero"] for row in rows),
        "parameters_restored": all(np.array_equal(model.params[name], original) for name, original in originals.items()),
    }
    return rows, detail
