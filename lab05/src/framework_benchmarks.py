"""CPU execution and tensor-layout experiments with measured preparation costs.

TorchScript tracing is the graph execution alternative in this experiment.
All reported execution times are milliseconds per call, amortized over an
explicit loop after warmup; no speedup direction is assumed.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _timing_settings(settings: dict[str, Any], defaults: dict[str, int]) -> dict[str, int]:
    return {
        key: _integer(settings.get(key, default), key, 0 if key == "warmup" else 1)
        for key, default in defaults.items()
    }


def _tensor_info(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "is_contiguous": bool(tensor.is_contiguous()),
    }


def _measure(function: Callable[[], torch.Tensor], settings: dict[str, int]) -> dict[str, Any]:
    """Warm up and time calls without tracking autograd."""
    times_ms = []
    with torch.no_grad():
        for _ in range(settings["warmup"]):
            function()
        for _ in range(settings["repeats"]):
            started = time.perf_counter_ns()
            for _ in range(settings["iterations"]):
                function()
            elapsed_ns = time.perf_counter_ns() - started
            times_ms.append(float(elapsed_ns / settings["iterations"] / 1_000_000))
    return {"times_ms": times_ms, "median_ms": float(statistics.median(times_ms))}


def _matching(left: torch.Tensor, right: torch.Tensor, name: str) -> dict[str, Any]:
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    matches = finite and bool(torch.allclose(left, right, rtol=1e-12, atol=1e-12))
    if not matches:
        raise RuntimeError(f"{name} produced nonfinite or numerically different outputs.")
    return {
        "allclose": True,
        "rtol": 1e-12,
        "atol": 1e-12,
        "max_abs_error": float((left - right).abs().max().item()),
    }


def _autograd_graph(model: torch.nn.Module, sample: torch.Tensor) -> tuple[dict[str, Any], str]:
    """Traverse an eager differentiable forward without calling backward.

    Strong references to every visited Function are retained throughout traversal
    so Python object identifiers cannot be recycled into incorrect graph edges.
    """
    with torch.enable_grad():
        output = model(sample)
    root = output.grad_fn
    if root is None:
        raise ValueError("The autograd example needs model parameters with requires_grad=True.")
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    references = [root]
    identifiers = {id(root): "n0"}
    nodes, edges = [], []
    position = 0
    while position < len(references):
        function = references[position]
        position += 1
        identifier = identifiers[id(function)]
        node = {"id": identifier, "operation": type(function).__name__}
        variable = getattr(function, "variable", None)
        if variable is not None:
            node["parameter"] = parameter_names.get(id(variable), "unnamed_leaf")
            node["tensor"] = _tensor_info(variable)
        nodes.append(node)
        for parent, input_nr in function.next_functions:
            if parent is None:
                continue
            if id(parent) not in identifiers:
                identifiers[id(parent)] = f"n{len(identifiers)}"
                references.append(parent)
            edges.append({
                "from": identifier,
                "to": identifiers[id(parent)],
                "input_nr": int(input_nr),
            })
    detail = {
        "root_id": "n0",
        "output_requires_grad": bool(output.requires_grad),
        "backward_executed": False,
        "nodes": nodes,
        "edges": edges,
    }
    lines = [
        "EAGER AUTOGRAD GRAPH",
        "Differentiable forward; next_functions edges point toward inputs/leaves.",
        "No backward call is made during this inspection.",
        "",
        "Nodes:",
    ]
    for node in nodes:
        parameter = f" parameter={node['parameter']}" if "parameter" in node else ""
        lines.append(f"{node['id']}: {node['operation']}{parameter}")
    lines += ["", "Edges:"]
    lines += [f"{edge['from']} -> {edge['to']} (input_nr={edge['input_nr']})" for edge in edges]
    return detail, "\n".join(lines)


def _row(experiment: str, variant: str, tensor: torch.Tensor, timing: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "variant": variant,
        "median_ms": timing["median_ms"],
        "shape": "x".join(str(size) for size in tensor.shape),
        "dtype": str(tensor.dtype),
        "contiguous": bool(tensor.is_contiguous()),
    }


def run_framework_benchmarks(
    model: torch.nn.Module, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Compare eager/traced inference and value-identical tensor layouts on CPU.

    The supplied trained model must already be in eval mode with CPU float64
    parameters. Seeds use local generators; parameters, gradients, model mode,
    global RNG state and thread settings are left unchanged.
    """
    parameters = list(model.parameters())
    if model.training:
        raise ValueError("Put the trained model in eval mode before benchmarking.")
    if not parameters or any(parameter.device.type != "cpu" or parameter.dtype != torch.float64 for parameter in parameters):
        raise ValueError("Framework benchmarks require a CPU float64 model.")

    execution_config = config.get("execution", {})
    execution_settings = _timing_settings(
        execution_config, {"warmup": 100, "iterations": 1000, "repeats": 5}
    )
    batches = execution_config.get("batches", [1, 64])
    if not isinstance(batches, (list, tuple)) or not batches:
        raise ValueError("execution.batches must be a nonempty list of batch sizes.")
    batches = [_integer(batch, "batch size", 1) for batch in batches]
    if len(set(batches)) != len(batches):
        raise ValueError("execution.batches must contain unique batch sizes.")
    execution_seed = _integer(execution_config.get("seed", 42), "execution.seed", 0)
    generator = torch.Generator(device="cpu").manual_seed(execution_seed)
    samples = [torch.randn(batch, 2, dtype=torch.float64, generator=generator) for batch in batches]

    # Tracing/checking and freezing are one-time preparation, outside all timed
    # inference loops. The same frozen graph is reused for every batch size.
    with torch.no_grad():
        started = time.perf_counter_ns()
        traced = torch.jit.trace(model, samples[0], check_trace=True)
        traced_at = time.perf_counter_ns()
        frozen = torch.jit.freeze(traced)
        frozen_at = time.perf_counter_ns()
    trace_ms = float((traced_at - started) / 1_000_000)
    freeze_ms = float((frozen_at - traced_at) / 1_000_000)
    preparation_ms = trace_ms + freeze_ms
    autograd_detail, autograd_text = _autograd_graph(model, samples[0])

    execution_detail = {
        "method": "torch.jit.trace followed by torch.jit.freeze; inference uses no_grad",
        "device": "cpu",
        "dtype": "torch.float64",
        "seed": execution_seed,
        **execution_settings,
        "timing_unit": "milliseconds per call, amortized across iterations",
        "trace_ms": trace_ms,
        "freeze_ms": freeze_ms,
        "preparation_ms": preparation_ms,
        "preparation_included_in_inference_times": False,
        "batches": [],
    }
    rows = []
    for batch, sample in zip(batches, samples):
        with torch.no_grad():
            eager_result, traced_result = model(sample), frozen(sample)
            correctness = _matching(eager_result, traced_result, "Eager and traced inference")
        eager_timing = _measure(lambda: model(sample), execution_settings)
        traced_timing = _measure(lambda: frozen(sample), execution_settings)
        saving_ms = eager_timing["median_ms"] - traced_timing["median_ms"]
        execution_detail["batches"].append({
            "batch_size": batch,
            "input": _tensor_info(sample),
            "output": _tensor_info(eager_result),
            "correctness": correctness,
            "eager": eager_timing,
            "torchscript_trace_frozen": traced_timing,
            "eager_over_traced_speedup": float(eager_timing["median_ms"] / traced_timing["median_ms"]),
            "break_even_calls": int(math.ceil(preparation_ms / saving_ms)) if saving_ms > 0 else None,
            "break_even_note": "Preparation amortization estimate from measured medians; undefined when traced inference is not faster.",
        })
        rows += [
            _row("execution", "eager", sample, eager_timing),
            _row("execution", "torchscript_trace_frozen", sample, traced_timing),
        ]

    tensor_config = config.get("tensors", {})
    tensor_settings = _timing_settings(
        tensor_config, {"warmup": 30, "iterations": 200, "repeats": 5}
    )
    tensor_seed = _integer(tensor_config.get("seed", 43), "tensors.seed", 0)
    tensor_rows = _integer(tensor_config.get("rows", 512), "tensors.rows", 1)
    tensor_columns = _integer(tensor_config.get("columns", 512), "tensors.columns", 2)
    tensor_generator = torch.Generator(device="cpu").manual_seed(tensor_seed)
    storage = torch.randn(tensor_rows, tensor_columns * 2, dtype=torch.float64, generator=tensor_generator)
    view = storage[:, ::2]
    contiguous = view.contiguous()
    if not torch.equal(view, contiguous):
        raise RuntimeError("The contiguous copy changed tensor values.")
    with torch.no_grad():
        strided_result = view.sum(dim=1)
        contiguous_result = contiguous.sum(dim=1)
        copied_result = view.contiguous().sum(dim=1)
        layout_correctness = _matching(strided_result, contiguous_result, "Tensor-layout reduction")
        copy_correctness = _matching(strided_result, copied_result, "Copy-then-reduce")
    layout_functions = {
        "noncontiguous_sum": lambda: view.sum(dim=1),
        "contiguous_sum": lambda: contiguous.sum(dim=1),
        "copy_then_sum": lambda: view.contiguous().sum(dim=1),
    }
    layout_timings = {
        variant: _measure(function, tensor_settings)
        for variant, function in layout_functions.items()
    }
    for variant, timing in layout_timings.items():
        rows.append(_row(
            "tensor_layout", variant, contiguous if variant == "contiguous_sum" else view, timing
        ))
    tensors_detail = {
        "seed": tensor_seed,
        **tensor_settings,
        "timing_unit": "milliseconds per call, amortized across iterations",
        "operation": "sum(dim=1)",
        "storage": _tensor_info(storage),
        "noncontiguous": _tensor_info(view),
        "contiguous": _tensor_info(contiguous),
        "same_input_values": True,
        "correctness": layout_correctness,
        "copy_then_sum_correctness": copy_correctness,
        "measurements": layout_timings,
        "noncontiguous_over_contiguous_speedup": float(layout_timings["noncontiguous_sum"]["median_ms"] / layout_timings["contiguous_sum"]["median_ms"]),
        "copy_then_sum_includes_allocation_and_copy": True,
        "layout_note": "The contiguous input is copied once before reduction timing; copy_then_sum allocates and copies on every call. Direction and magnitude come from actual measurements.",
    }
    frozen_graph = frozen.inlined_graph
    detail = {
        "torch_version": str(torch.__version__),
        "torch_num_threads": int(torch.get_num_threads()),
        "execution": execution_detail,
        "tensors": tensors_detail,
        "graphs": {
            "autograd": autograd_detail,
            "traced_frozen": {
                "node_count": len(list(frozen_graph.nodes())),
                "operations": [node.kind() for node in frozen_graph.nodes()],
                "trace_example": _tensor_info(samples[0]),
            },
        },
    }
    graph_text = f"{autograd_text}\n\nFROZEN TRACED INLINED GRAPH\n{frozen_graph}\n"
    return rows, detail, graph_text
