import numpy as np
import pandas as pd
import pytest

from power_quant_strategies.utils.evaluate_model.evaluate_classification_metrics import evaluate_classification_metrics


def test_perfect_predictions():
    y_true = pd.DataFrame({"target": [0, 0, 1, 1]})
    y_pred = pd.DataFrame({"preds": [0.1, 0.2, 0.8, 0.9]})

    metrics = evaluate_classification_metrics(y_true, y_pred, split="test", save_path=None)["Value"]

    assert metrics["accuracy_score"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc_score"] == 1.0
    assert metrics["cm_true_positives"] == 2
    assert metrics["cm_true_negatives"] == 2
    assert metrics["cm_false_positives"] == 0
    assert metrics["zero_one_loss"] == 0.0


def test_threshold_decides_the_predicted_class():
    y_true = pd.DataFrame({"target": [0, 1]})
    y_pred = pd.DataFrame({"preds": [0.3, 0.4]})

    at_half = evaluate_classification_metrics(y_true, y_pred, split="test", threshold=0.5, save_path=None)["Value"]
    at_035 = evaluate_classification_metrics(y_true, y_pred, split="test", threshold=0.35, save_path=None)["Value"]

    assert at_half["accuracy_score"] == 0.5
    assert at_035["accuracy_score"] == 1.0


def test_sample_weight_is_used():
    y_true = pd.DataFrame({"target": [0, 1]})
    y_pred = pd.DataFrame({"preds": [0.1, 0.1]})  # second one is wrong
    weight = pd.DataFrame({"w": [9.0, 1.0]})

    metrics = evaluate_classification_metrics(y_true, y_pred, split="test", sample_weight=weight, save_path=None)["Value"]

    assert metrics["accuracy_score"] == pytest.approx(0.9)


def test_undefined_metrics_are_nan():
    # a single class makes roc_auc undefined
    metrics = evaluate_classification_metrics(pd.DataFrame({"t": [1, 1]}), pd.DataFrame({"p": [0.7, 0.8]}), split="test", save_path=None)["Value"]
    assert np.isnan(metrics["roc_auc_score"])
    assert metrics["accuracy_score"] == 1.0


def test_saves_sorted_metrics(results_dir):
    metrics = evaluate_classification_metrics(pd.DataFrame({"t": [0, 1]}), pd.DataFrame({"p": [0.2, 0.7]}), split="train", save_path=results_dir)
    assert list(metrics.index) == sorted(metrics.index)
    saved = pd.read_parquet(f"{results_dir}/03_evaluate_model/classification_metrics_train.parquet")
    pd.testing.assert_frame_equal(saved, metrics)


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        (pd.Series([0, 1]), pd.DataFrame({"p": [0.1, 0.9]})),
        (pd.DataFrame({"t": [0, 1]}), pd.DataFrame({"p": [0.1, 0.9, 0.5]})),
    ],
)
def test_invalid_input(y_true, y_pred):
    with pytest.raises(AssertionError):
        evaluate_classification_metrics(y_true, y_pred, split="test", save_path=None)
