import logging
import os
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    class_likelihood_ratios,
    cohen_kappa_score,
    confusion_matrix,
    d2_log_loss_score,
    f1_score,
    hamming_loss,
    hinge_loss,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    zero_one_loss,
)


def evaluate_classification_metrics(  # noqa: C901
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    split: str,
    sample_weight: pd.DataFrame = None,
    threshold: float = 0.5,
    save_path: Union[str, None] = os.getcwd(),
) -> pd.DataFrame:
    """
    Calculate all available binary classification metrics from scikit-learn.

    :param y_true: Ground truth binary labels.
    :param y_pred: Predicted binary labels or probabilities.
    :param split: Defines the split (train, valid, test)
    :param sample_weight: Optional sample weights.
    :param threshold: Optional threshold.
    :param save_path: Directory to save the metrics
    :return: Classification metrics with metric names as the index and their values in a column named "Value".
    """
    # give user feedback
    logging.info("evaluate classification metrics")

    # check inputs
    assert isinstance(y_true, pd.DataFrame), "y_true must be a pd.DataFrame"
    assert isinstance(y_pred, pd.DataFrame), "y_pred must be a pd.DataFrame"
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"

    # make sure y_true and y_pred are series as scikit-learn expects that format
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    if sample_weight is not None:
        sample_weight = sample_weight.squeeze()

    # save probabilities as some metrics want probabilities while others want class indicator
    y_proba = y_pred

    # ensure integer values
    y_true = y_true >= threshold
    y_pred = y_pred >= threshold

    # initialize metrics dict to be later transformed to a pd.DataFrame
    metrics = {}

    # calculate metrics
    try:
        metrics["accuracy_score"] = accuracy_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    except (IndexError, ValueError, ZeroDivisionError):
        metrics["accuracy_score"] = np.nan
    try:
        metrics["average_precision_score"] = average_precision_score(y_true=y_true, y_score=y_proba, sample_weight=sample_weight)
    except (IndexError, ValueError, ZeroDivisionError):
        metrics["average_precision_score"] = np.nan
    try:
        metrics["balanced_accuracy_score"] = balanced_accuracy_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    except (IndexError, ValueError, ZeroDivisionError):
        metrics["balanced_accuracy_score"] = np.nan
    try:
        metrics["brier_score_loss"] = brier_score_loss(y_true=y_true, y_proba=y_proba, sample_weight=sample_weight)
    except (IndexError, ValueError, ZeroDivisionError):
        metrics["brier_score_loss"] = np.nan
    try:
        positive_likelihood_ratio, negative_likelihood_ratio = class_likelihood_ratios(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
        metrics["class_likelihood_ratios_positive"] = positive_likelihood_ratio
        metrics["class_likelihood_ratios_negative"] = negative_likelihood_ratio
    except (IndexError, ValueError, ZeroDivisionError):
        metrics["class_likelihood_ratios_positive"] = np.nan
        metrics["class_likelihood_ratios_negative"] = np.nan
    try:
        metrics["cohen_kappa_score"] = cohen_kappa_score(y1=y_true, y2=y_pred, sample_weight=sample_weight)
    except (IndexError, ValueError, ZeroDivisionError):
        metrics["cohen_kappa_score"] = np.nan
    try:
        true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1], sample_weight=sample_weight).ravel()
        metrics["cm_false_negatives"] = false_negative
        metrics["cm_false_positives"] = false_positive
        metrics["cm_true_negatives"] = true_negative
        metrics["cm_true_positives"] = true_positive
    except (IndexError, ValueError, ZeroDivisionError):
        metrics["cm_false_negatives"] = np.nan
        metrics["cm_false_positives"] = np.nan
        metrics["cm_true_negatives"] = np.nan
        metrics["cm_true_positives"] = np.nan
    try:
        metrics["d2_log_loss_score"] = d2_log_loss_score(y_true=y_true, y_pred=y_proba, sample_weight=sample_weight)
    except (IndexError, ValueError, ZeroDivisionError):
        metrics["d2_log_loss_score"] = np.nan
    try:
        metrics["f1_score"] = f1_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    except (IndexError, ValueError, ZeroDivisionError):
        metrics["f1_score"] = np.nan
    try:
        metrics["hamming_loss"] = hamming_loss(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    except (IndexError, ValueError, ZeroDivisionError):
        metrics["hamming_loss"] = np.nan
    try:
        metrics["hinge_loss"] = hinge_loss(y_true=y_true, pred_decision=y_proba, sample_weight=sample_weight)
    except (IndexError, ValueError, ZeroDivisionError):
        metrics["hinge_loss"] = np.nan
    try:
        metrics["jaccard_score"] = jaccard_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    except (IndexError, ValueError, ZeroDivisionError):
        metrics["jaccard_score"] = np.nan
    try:
        metrics["log_loss"] = log_loss(y_true=y_true, y_pred=y_proba, sample_weight=sample_weight)
    except (IndexError, ValueError, ZeroDivisionError):
        metrics["log_loss"] = np.nan
    try:
        metrics["matthews_corrcoef"] = matthews_corrcoef(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    except (IndexError, ValueError, ZeroDivisionError):
        metrics["matthews_corrcoef"] = np.nan
    try:
        metrics["precision_score"] = precision_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    except (IndexError, ValueError, ZeroDivisionError):
        metrics["precision_score"] = np.nan
    try:
        metrics["recall_score"] = recall_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    except (IndexError, ValueError, ZeroDivisionError):
        metrics["recall_score"] = np.nan
    try:
        metrics["roc_auc_score"] = roc_auc_score(y_true=y_true, y_score=y_proba, sample_weight=sample_weight)
    except (IndexError, ValueError, ZeroDivisionError):
        metrics["roc_auc_score"] = np.nan
    try:
        metrics["zero_one_loss"] = zero_one_loss(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    except (IndexError, ValueError, ZeroDivisionError):
        metrics["zero_one_loss"] = np.nan

    # convert metrics to DataFrame with index as metric names and column as "Value"
    metrics = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])

    # Sort DataFrame alphabetically by metric name
    metrics.sort_index(inplace=True)

    # save metrics to disk
    if save_path is not None:
        metrics.to_parquet(path=os.path.join(save_path, "03_evaluate_model", f"classification_metrics_{split}.parquet"), index=True)
        metrics.to_excel(excel_writer=os.path.join(save_path, "03_evaluate_model", f"classification_metrics_{split}.xlsx"))

    # return classification metrics with metric names as the index and their values in a column named "Value"
    return metrics
