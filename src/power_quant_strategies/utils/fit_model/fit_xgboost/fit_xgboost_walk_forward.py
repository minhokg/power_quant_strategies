import logging
import os
from pathlib import Path
from typing import Union

import pandas as pd
from joblib import Parallel, delayed

from power_quant_strategies.utils.evaluate_model.check_model_score_vs_spread_dependency import check_model_score_vs_spread_dependency
from power_quant_strategies.utils.evaluate_model.evaluate_classification_metrics import evaluate_classification_metrics
from power_quant_strategies.utils.evaluate_model.evaluate_trading_metrics import evaluate_trading_metrics
from power_quant_strategies.utils.fit_model.calculate_spread_realized_and_expected import add_spread_realized_and_expected
from power_quant_strategies.utils.fit_model.find_optimal_threshold_and_maximum_pnl import find_optimal_threshold_and_maximum_pnl
from power_quant_strategies.utils.fit_model.fit_xgboost.fit_xgboost import fit_xgboost
from power_quant_strategies.utils.helper.get_usable_cpu_count import get_usable_cpu_count


def _process_walk_forward_period(
    i_days_walk_forward: int,
    n_days_walk_forward: int,
    data: pd.DataFrame,
    column_name_target: str,
    hyperparameter: dict,
    shuffle: bool = True,
    percentage_train: float = 0.8,
    n_days_sliding_window: int = 183,
    n_estimators: int = 500,
    early_stopping_rounds: int = 10,
    verbose: Union[bool, int, None] = False,
    random_state: int = 42,
    device: str = "cpu",
):
    # get unique days (normalized datetime) from the index
    days = data.index.normalize().unique()

    # define the sliding window for training + validation based on days
    start_day_index = i_days_walk_forward
    end_day_index = i_days_walk_forward + n_days_sliding_window
    current_days = days[start_day_index:end_day_index]

    # select the rows belonging to the current sliding window of days
    data_train_valid = data[data.index.normalize().isin(current_days)].copy()

    # optionally shuffle the train/validation set
    if shuffle:
        data_train_valid = data_train_valid.sample(frac=1, random_state=random_state)

    # split the train/validation set based on the percentage for training
    end_train = int(len(data_train_valid) * percentage_train)
    data_train = data_train_valid.iloc[:end_train].copy()
    data_valid = data_train_valid.iloc[end_train:].copy()

    # define the test window as the days immediately following the sliding window
    test_days = days[end_day_index : end_day_index + 1]
    data_test = data[data.index.normalize().isin(test_days)].copy()

    # give user feedback with
    logging.info(
        f"Worker {os.getpid()} processing task {i_days_walk_forward} / {n_days_walk_forward} Train+Valid: {data_train.index[0]} - {data_valid.index[-1]} \
    ({len(data_train_valid)}) / Test: {data_test.index[0]} - {data_test.index[-1]} ({len(data_test)} timesteps)"
    )

    # fit model
    model, data_train, data_valid, data_test = fit_xgboost(
        data_train=data_train,
        data_valid=data_valid,
        data_test=data_test,
        column_name_target=column_name_target,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
        hyperparameter=hyperparameter,
        random_state=random_state,
        device=device,
    )

    # find optimal threshold and maximum pnl
    threshold_train, maximum_pnl_train = find_optimal_threshold_and_maximum_pnl(
        data=data_train,
        column_name_target=column_name_target,
        split="train",
        save_path=None,
    )
    threshold_valid, maximum_pnl_valid = find_optimal_threshold_and_maximum_pnl(
        data=data_valid,
        column_name_target=column_name_target,
        split="valid",
        save_path=None,
    )
    threshold_test, maximum_pnl_test = find_optimal_threshold_and_maximum_pnl(
        data=data_test,
        column_name_target=column_name_target,
        split="test",
        save_path=None,
    )

    # evaluate classification metrics
    classification_metrics_train = evaluate_classification_metrics(
        y_true=data_train[[column_name_target]], y_pred=data_train[["preds"]], sample_weight=data_train[["sample_weight"]], threshold=threshold_train, split="train", save_path=None
    )
    classification_metrics_valid = evaluate_classification_metrics(
        y_true=data_valid[[column_name_target]], y_pred=data_valid[["preds"]], sample_weight=data_valid[["sample_weight"]], threshold=threshold_valid, split="valid", save_path=None
    )
    classification_metrics_test = evaluate_classification_metrics(
        y_true=data_test[[column_name_target]], y_pred=data_test[["preds"]], sample_weight=data_test[["sample_weight"]], threshold=threshold_test, split="test", save_path=None
    )

    # trading metrics
    trading_metrics_train = evaluate_trading_metrics(data=data_train, column_name_target=column_name_target, split="train", threshold=threshold_train, save_path=None)
    trading_metrics_valid = evaluate_trading_metrics(data=data_valid, column_name_target=column_name_target, split="valid", threshold=threshold_valid, save_path=None)
    trading_metrics_test = evaluate_trading_metrics(data=data_test, column_name_target=column_name_target, split="test", threshold=threshold_test, save_path=None)

    threshold_train_valid, _ = find_optimal_threshold_and_maximum_pnl(data=pd.concat(objs=[data_train, data_valid], axis=0), column_name_target=column_name_target, split="train_valid", save_path=None)

    data_train, data_valid, data_test = add_spread_realized_and_expected(
        data_train=data_train,
        data_valid=data_valid,
        data_test=data_test,
        column_name_target=column_name_target,
    )
    data_test["threshold_train_valid"] = threshold_train_valid
    data_test["threshold_test"] = threshold_test

    return (
        data_test,
        threshold_train,
        maximum_pnl_train,
        threshold_valid,
        maximum_pnl_valid,
        threshold_test,
        maximum_pnl_test,
        classification_metrics_train,
        classification_metrics_valid,
        classification_metrics_test,
        trading_metrics_train,
        trading_metrics_valid,
        trading_metrics_test,
    )


def fit_xgboost_walk_forward(
    data: pd.DataFrame,
    column_name_target: str,
    hyperparameter: dict,
    shuffle: bool = True,
    percentage_train: float = 0.8,
    n_days_sliding_window: int = 183,
    n_estimators: int = 500,
    early_stopping_rounds: int = 10,
    verbose: Union[bool, int, None] = False,
    random_state: int = 42,
    device: str = "cpu",
    save_path: str | None = str(Path.cwd()),
) -> tuple:
    """
    Train lstm with walk forward cross validation.

    :param data: Original data with features, target.
    :param column_name_target: Name of target column.
    :param hyperparameter: Hyperparameters for lstm.
    :param shuffle: Defines whether to shuffle train and valid before splitting in train and valid.
    :param percentage_train: Defines how much percent of the data will be used for train.
    :param n_estimators: Defines how many trees in the forest.
    :param early_stopping_rounds: Defines how many rounds of early stopping.
    :param verbose: Defines whether to display the progress of early stopping.
    :param n_days_sliding_window: Number of days used for training
    :param random_state: Random seed for shuffle and lstm.
    :param device: Device to use for XGBoost training
    :return:
    """
    # give user feedback
    logging.info("fit lstm walk forward")

    # make sure there is enough data that matches the user settings
    assert len(data.index.normalize().unique()) > n_days_sliding_window

    # make sure data index is sorted to not leak future information
    data = data.sort_index()

    # determine how many walk forward periods there are
    n_days_walk_forward = len(data.index.normalize().unique()) - n_days_sliding_window

    # run walk forward in parallel to speed things up
    results = Parallel(n_jobs=get_usable_cpu_count())(
        delayed(_process_walk_forward_period)(
            i, n_days_walk_forward, data, column_name_target, hyperparameter, shuffle, percentage_train, n_days_sliding_window, n_estimators, early_stopping_rounds, verbose, random_state, device
        )
        for i in range(n_days_walk_forward)
    )

    # unpack results
    data_test_all = pd.concat([res[0] for res in results], axis=0)
    threshold_train_all = [res[1] for res in results]
    maximum_pnl_train_all = [res[2] for res in results]
    threshold_valid_all = [res[3] for res in results]
    maximum_pnl_valid_all = [res[4] for res in results]
    threshold_test_all = [res[5] for res in results]
    maximum_pnl_test_all = [res[6] for res in results]
    classification_metrics_train_all = pd.concat([res[7] for res in results], axis=1)
    classification_metrics_valid_all = pd.concat([res[8] for res in results], axis=1)
    classification_metrics_test_all = pd.concat([res[9] for res in results], axis=1)
    trading_metrics_train_all = pd.concat([res[10] for res in results], axis=1)
    trading_metrics_valid_all = pd.concat([res[11] for res in results], axis=1)
    trading_metrics_test_all = pd.concat([res[12] for res in results], axis=1)

    # combine all thresholds and convert to pd.DataFrame
    threshold_all = {
        "threshold_train_all": threshold_train_all,
        "threshold_valid_all": threshold_valid_all,
        "threshold_test_all": threshold_test_all,
    }
    threshold_all = pd.DataFrame(threshold_all)

    # combine all maximum_pnl and convert to pd.DataFrame
    maximum_pnl_all = {
        "maximum_pnl_train_all": maximum_pnl_train_all,
        "maximum_pnl_valid_all": maximum_pnl_valid_all,
        "maximum_pnl_test_all": maximum_pnl_test_all,
    }
    maximum_pnl_all = pd.DataFrame(maximum_pnl_all)

    # check model score vs spread dependency
    check_model_score_vs_spread_dependency(
        data=data_test_all,
        column_name_target=column_name_target,
        split="test",
        save_path=save_path,
    )

    # get last year threshold and maximum pnl
    data_copy = data_test_all.copy()
    data_last_year = data_copy[data_copy.index >= (data_copy.index.max() - pd.DateOffset(years=1))]
    threshold_test, maximum_pnl_test = find_optimal_threshold_and_maximum_pnl(
        data=data_last_year,
        column_name_target=column_name_target,
        split="test",
        save_path=save_path,
        postfix="last_year",
    )

    # classification metrics
    classification_metrics_test = evaluate_classification_metrics(
        y_true=data_test_all[[column_name_target]],
        y_pred=data_test_all[["preds"]],
        sample_weight=data_test_all[["sample_weight"]],
        threshold=threshold_all[["threshold_train_all", "threshold_valid_all"]].mean().mean(),
        split="test",
        save_path=save_path,
    )

    # trading metrics
    trading_metrics_test = evaluate_trading_metrics(
        data=data_test_all,
        column_name_target=column_name_target,
        split="test",
        threshold=threshold_all[["threshold_train_all", "threshold_valid_all"]].mean().mean(),
        save_path=save_path,
    )

    # return concatenated test data, all thresholds, all maximum pnl and all metrics
    return (
        data_test_all,
        threshold_all,
        maximum_pnl_all,
        classification_metrics_train_all,
        classification_metrics_valid_all,
        classification_metrics_test_all,
        trading_metrics_train_all,
        trading_metrics_valid_all,
        trading_metrics_test_all,
        threshold_test,
        maximum_pnl_test,
        classification_metrics_test,
        trading_metrics_test,
    )
