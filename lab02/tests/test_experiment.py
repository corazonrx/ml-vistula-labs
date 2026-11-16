"""Check digits preparation, held-out-label isolation and softmax validation."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lab02.src.experiment import parameter_hash, prepare_data, probability_check, train_model


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[1] / "config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_digits_preparation_preserves_exact_pixels_labels_and_stratified_source_ids(config):
    raw = load_digits()
    assert raw.data.shape == (1797, 64)
    data, detail = prepare_data(config["data"])
    train_ids, test_ids = detail["train_source_ids"], detail["test_source_ids"]
    expected_train, expected_test = train_test_split(
        np.arange(1797), test_size=0.3, stratify=raw.target, random_state=42
    )

    assert len(train_ids) == detail["train_rows"] == 1257
    assert len(test_ids) == detail["test_rows"] == 540
    assert set(train_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(test_ids) == set(range(1797))
    np.testing.assert_array_equal(train_ids, expected_train)
    np.testing.assert_array_equal(test_ids, expected_test)
    assert detail["input_range"] == [0.0, 1.0]
    assert detail["classes"] == list(range(10))
    for partition, source_ids in [("train", train_ids), ("test", test_ids)]:
        inputs, labels, one_hot = (data[f"{partition}_{name}"] for name in ["x", "labels", "y"])
        assert inputs.dtype == one_hot.dtype == np.float64
        assert inputs.shape == (len(source_ids), 64)
        assert one_hot.shape == (len(source_ids), 10)
        assert np.all((0 <= inputs) & (inputs <= 1))
        np.testing.assert_array_equal(inputs, raw.data[source_ids] / 16.0)
        np.testing.assert_array_equal(labels, raw.target[source_ids])
        np.testing.assert_array_equal(one_hot, np.eye(10)[labels])
        np.testing.assert_array_equal(one_hot.sum(axis=1), np.ones(len(source_ids)))
        assert set(labels) == set(range(10))
        assert detail[f"{partition}_class_counts"] == np.bincount(labels, minlength=10).tolist()


def test_remainder_batch_updates_and_training_are_independent_of_heldout_labels(config):
    settings = copy.deepcopy(config)
    settings["training"]["epochs"] = 2
    prepared, _ = prepare_data(settings["data"])
    small = {
        key: values[:41 if key.startswith("train_") else 11].copy()
        for key, values in prepared.items()
    }
    changed = {key: values.copy() for key, values in small.items()}
    changed["test_labels"] = (changed["test_labels"] + 1) % 10
    changed["test_y"] = np.eye(10)[changed["test_labels"]]
    assert not np.array_equal(small["test_labels"], changed["test_labels"])
    np.testing.assert_array_equal(small["test_x"], changed["test_x"])

    original_model, original_row, original_history, original_detail = train_model(
        small, width=16, learning_rate=0.1, config=settings
    )
    changed_model, changed_row, changed_history, changed_detail = train_model(
        changed, width=16, learning_rate=0.1, config=settings
    )

    assert original_detail["total_updates"] == changed_detail["total_updates"] == 4
    assert original_detail["batches_per_epoch"] == 2
    assert original_detail["last_batch_size"] == 9
    assert original_history == changed_history
    assert len(original_history) == 2
    assert original_row["final_train_loss"] == changed_row["final_train_loss"]
    assert original_row["final_train_accuracy"] == changed_row["final_train_accuracy"]
    assert original_detail["initial_parameters_sha256"] != original_detail["final_parameters_sha256"]
    assert original_detail["final_parameters_sha256"] == changed_detail["final_parameters_sha256"]
    assert parameter_hash(original_model) == parameter_hash(changed_model)
    for name in original_model.params:
        np.testing.assert_array_equal(original_model.params[name], changed_model.params[name])


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        (np.full((3, 9), 1 / 9), "Expected"),
        (np.full((3, 10), np.nan), "finite"),
        (np.full((3, 10), 0.05), "row-sum"),
    ],
    ids=["wrong_class_shape", "nonfinite_values", "incorrect_row_sums"],
)
def test_probability_check_rejects_invalid_outputs(invalid, message):
    assert probability_check(np.full((3, 10), 0.1)) < 1e-6
    with pytest.raises(AssertionError, match=message):
        probability_check(invalid)
