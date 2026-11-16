"""Check experiment correctness and state preservation, not speed direction."""

import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lab05.src.framework_benchmarks import run_framework_benchmarks


@pytest.fixture(scope="module")
def experiment():
    torch.manual_seed(24)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1), torch.nn.Sigmoid()
    ).double().eval()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    before_gradients = [parameter.grad.clone() for parameter in model.parameters()]
    before_rng = torch.get_rng_state().clone()
    before_threads = torch.get_num_threads()
    rows, detail, graph = run_framework_benchmarks(model, {
        "execution": {"seed": 42, "batches": [1, 4], "warmup": 1, "iterations": 2, "repeats": 2},
        "tensors": {"seed": 43, "rows": 8, "columns": 8, "warmup": 1, "iterations": 2, "repeats": 2},
    })
    return {
        "model": model, "rows": rows, "detail": detail, "graph": graph,
        "before_parameters": before_parameters, "before_gradients": before_gradients,
        "before_rng": before_rng, "after_rng": torch.get_rng_state().clone(),
        "before_threads": before_threads, "after_threads": torch.get_num_threads(),
    }


def test_timings_cover_both_batches_and_layouts_with_finite_per_call_values(experiment):
    rows, detail = experiment["rows"], experiment["detail"]
    assert len(rows) == 7
    assert {(row["variant"], row["shape"]) for row in rows if row["experiment"] == "execution"} == {
        (variant, shape)
        for variant in ["eager", "torchscript_trace_frozen"]
        for shape in ["1x2", "4x2"]
    }
    assert all(math.isfinite(row["median_ms"]) and row["median_ms"] > 0 for row in rows)
    assert all(row["dtype"] == "torch.float64" for row in rows)
    execution = detail["execution"]
    assert execution["warmup"] == 1 and execution["iterations"] == 2
    assert execution["preparation_ms"] >= 0
    assert execution["preparation_included_in_inference_times"] is False
    for batch in execution["batches"]:
        assert batch["correctness"]["allclose"] is True
        assert batch["output"]["shape"] == [batch["batch_size"], 1]
        for variant in ["eager", "torchscript_trace_frozen"]:
            assert len(batch[variant]["times_ms"]) == 2
        if batch["break_even_calls"] is not None:
            assert batch["break_even_calls"] > 0
    # Entire evidence structure must be writable as portable JSON/YAML data.
    json.dumps(detail, allow_nan=False)


def test_same_value_layout_experiment_checks_strides_and_numerical_agreement(experiment):
    tensors = experiment["detail"]["tensors"]
    assert tensors["storage"]["shape"] == [8, 16]
    assert tensors["noncontiguous"]["shape"] == tensors["contiguous"]["shape"] == [8, 8]
    assert tensors["noncontiguous"]["stride"] == [16, 2]
    assert tensors["contiguous"]["stride"] == [8, 1]
    assert tensors["noncontiguous"]["is_contiguous"] is False
    assert tensors["contiguous"]["is_contiguous"] is True
    assert tensors["same_input_values"] is True
    assert tensors["correctness"]["allclose"] is True
    assert tensors["copy_then_sum_correctness"]["allclose"] is True
    assert set(tensors["measurements"]) == {"noncontiguous_sum", "contiguous_sum", "copy_then_sum"}
    assert all(len(timing["times_ms"]) == 2 for timing in tensors["measurements"].values())


def test_graph_inspection_and_benchmarks_preserve_model_gradients_and_rng(experiment):
    model = experiment["model"]
    assert model.training is False
    for parameter, before, gradient in zip(
        model.parameters(), experiment["before_parameters"], experiment["before_gradients"]
    ):
        assert torch.equal(parameter, before)
        assert torch.equal(parameter.grad, gradient)
    assert torch.equal(experiment["before_rng"], experiment["after_rng"])
    assert experiment["before_threads"] == experiment["after_threads"]
    autograd = experiment["detail"]["graphs"]["autograd"]
    assert autograd["output_requires_grad"] is True
    assert autograd["backward_executed"] is False
    operations = [node["operation"] for node in autograd["nodes"]]
    assert any(operation.startswith("SigmoidBackward") for operation in operations)
    assert "AccumulateGrad" in operations
    identifiers = {node["id"] for node in autograd["nodes"]}
    assert autograd["root_id"] in identifiers
    assert autograd["edges"]
    assert all(edge["from"] in identifiers and edge["to"] in identifiers for edge in autograd["edges"])
    assert experiment["detail"]["graphs"]["traced_frozen"]["node_count"] > 0
    assert "EAGER AUTOGRAD GRAPH" in experiment["graph"]
    assert "FROZEN TRACED INLINED GRAPH" in experiment["graph"]
    assert "graph(" in experiment["graph"]
