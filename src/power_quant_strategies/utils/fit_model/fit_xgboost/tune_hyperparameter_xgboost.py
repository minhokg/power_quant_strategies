import json
import os
import warnings
from functools import partial
from typing import Union

import joblib
import optuna
import pandas as pd

from power_quant_strategies.utils.evaluate_model.evaluate_classification_metrics import evaluate_classification_metrics
from power_quant_strategies.utils.evaluate_model.evaluate_trading_metrics import evaluate_trading_metrics
from power_quant_strategies.utils.fit_model.find_optimal_threshold_and_maximum_pnl import find_optimal_threshold_and_maximum_pnl
from power_quant_strategies.utils.fit_model.fit_xgboost.fit_xgboost import fit_xgboost
from power_quant_strategies.utils.helper.get_usable_cpu_count import get_usable_cpu_count


def tune_hyperparameter_xgboost(
    data_train: pd.DataFrame,
    data_valid: pd.DataFrame,
    data_test: pd.DataFrame,
    column_name_target: str,
    n_estimators: int = 500,
    early_stopping_rounds: int = 10,
    verbose: Union[bool, int, None] = True,
    random_state: int = 42,
    direction: str = "maximize",
    n_trials: int = 100,
    max_run_time_per_model_fit: float = None,
    show_progress_bar: bool = True,
    save_path: Union[str, None] = os.getcwd(),
    device: str = "cpu",
) -> tuple[optuna.Study, pd.DataFrame]:
    """
    Tune hyperparameters of XGBoost model.

    :param data_train: DataFrame of training data
    :param data_valid: DataFrame of validation data
    :param data_test: DataFrame of testing data
    :param column_name_target: Name of target column
    :param n_estimators: Number of estimators for fit_xgboost
    :param early_stopping_rounds: Number of early stopping rounds for fit_xgboost
    :param verbose: Defines how verbose fit_xgboost should be
    :param random_state: Random seed for fit_xgboost.
    :param direction: Defines whether to maximize or minimize
    :param n_trials: Defines how many hyperparameter tuning rounds
    :param max_run_time_per_model_fit: Defines how long one model fit could take to make sure backtesting still runs in\
    sensible time
    :param show_progress_bar: Defines whether to show progress bar
    :param save_path: Path to save the result
    :param device: Device to use for XGBoost training (e.g. "cpu", "cuda", "gpu")
    :return: Tuple of optuna Study object and pd.DataFrame containing the best hyperparameter values.
    """
    # define objective with data inputs
    objective_with_data = partial(
        _objective,
        data_train=data_train,
        data_valid=data_valid,
        data_test=data_test,
        column_name_target=column_name_target,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
        random_state=random_state,
        device=device,
    )

    # disable warnings
    warnings.filterwarnings("ignore")

    # create optuna sampler in order to fix the seed
    sampler = optuna.samplers.TPESampler(seed=random_state)

    # create an optuna study
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
    )

    # run optimization
    study.optimize(
        func=objective_with_data,
        n_trials=n_trials,
        n_jobs=get_usable_cpu_count(),
        show_progress_bar=show_progress_bar,
    )

    # enable warnings
    warnings.resetwarnings()

    # save all trial info to excel
    trials = study.trials_dataframe()

    # extract the best parameters
    trials_within_runtime_limits = trials[trials["duration"].dt.total_seconds() <= max_run_time_per_model_fit]
    if trials_within_runtime_limits.shape[0] == 0:
        best_params = pd.DataFrame.from_dict(study.trials[trials["duration"].dt.total_seconds().idxmin()].params, orient="index", columns=["Value"])
    else:
        best_params = pd.DataFrame.from_dict(study.trials[trials_within_runtime_limits["value"].idxmax()].params, orient="index", columns=["Value"])

    # save results to disk
    if save_path:
        # save the trials
        trials.to_parquet(path=os.path.join(save_path, "trials.xlsx"), index=True)
        trials.to_excel(excel_writer=os.path.join(save_path, "trials.xlsx"))

        # save study object instance
        joblib.dump(value=study, filename=os.path.join(save_path, "optuna_study.pkl"))

        # save best parameters as EXCEL
        best_params.to_excel(excel_writer=os.path.join(save_path, "best_params.xlsx"))

        # save best parameters as copy-pastable Json
        best_params_dict = best_params["Value"].to_dict()
        with open(os.path.join(save_path, "best_params.json"), "w") as f_json:
            json.dump(best_params_dict, f_json, indent=4)

    # return tuple of optuna Study object and pd.DataFrame containing the best hyperparameter values.
    return study, best_params


def _objective(
    trial,
    data_train: pd.DataFrame,
    data_valid: pd.DataFrame,
    data_test: pd.DataFrame,
    column_name_target: str,
    n_estimators: int = 500,
    early_stopping_rounds: int = 10,
    verbose: Union[bool, int, None] = True,
    random_state: int = 42,
    name_metric_to_optimize: str = "roc_auc_score",
    device: str = "cpu",
):
    # define the hyperparameter space for Optuna
    hyperparameter = {
        "eval_metric": "logloss",
        "eta": trial.suggest_loguniform("eta", 0.01, 0.3),
        "gamma": trial.suggest_loguniform("gamma", 1e-8, 10.0),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 1e-3, 10.0),
        "max_delta_step": trial.suggest_uniform("max_delta_step", 0.0, 10.0),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_uniform("colsample_bylevel", 0.5, 1.0),
        "lambda": trial.suggest_loguniform("lambda", 1e-3, 10.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-3, 10.0),
        "tree_method": "exact",
    }

    if hyperparameter["tree_method"] in ["hist", "approx"]:
        hyperparameter["colsample_bynode"] = trial.suggest_uniform("colsample_bynode", 0.5, 1.0)

    # fit single fit_xgboost
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
    )
    threshold_valid, maximum_pnl_valid = find_optimal_threshold_and_maximum_pnl(
        data=data_valid,
        column_name_target=column_name_target,
        split="valid",
    )
    threshold_test, maximum_pnl_test = find_optimal_threshold_and_maximum_pnl(
        data=data_test,
        column_name_target=column_name_target,
        split="test",
    )

    # evaluate classification metrics
    classification_metrics_train = evaluate_classification_metrics(
        y_true=data_train[[column_name_target]],
        y_pred=data_train[["preds"]],
        sample_weight=data_train[["sample_weight"]],
        threshold=threshold_train,
        split="train",
        save_path=None,
    )
    classification_metrics_valid = evaluate_classification_metrics(
        y_true=data_valid[[column_name_target]],
        y_pred=data_valid[["preds"]],
        sample_weight=data_valid[["sample_weight"]],
        threshold=threshold_valid,
        split="valid",
        save_path=None,
    )
    classification_metrics_test = evaluate_classification_metrics(
        y_true=data_test[[column_name_target]],
        y_pred=data_test[["preds"]],
        sample_weight=data_test[["sample_weight"]],
        threshold=threshold_test,
        split="test",
        save_path=None,
    )

    # evaluate_trading_metrics
    trading_metrics_train = evaluate_trading_metrics(
        data=data_train,
        column_name_target=column_name_target,
        split="train",
        threshold=threshold_train,
        save_path=None,
    )
    trading_metrics_valid = evaluate_trading_metrics(
        data=data_valid,
        column_name_target=column_name_target,
        split="valid",
        threshold=threshold_valid,
        save_path=None,
    )
    trading_metrics_test = evaluate_trading_metrics(
        data=data_test,
        column_name_target=column_name_target,
        split="test",
        threshold=threshold_test,
        save_path=None,
    )

    # merge metrics together
    metrics_train = pd.concat(objs=[classification_metrics_train, trading_metrics_train], axis=0)
    metrics_valid = pd.concat(objs=[classification_metrics_valid, trading_metrics_valid], axis=0)
    metrics_test = pd.concat(objs=[classification_metrics_test, trading_metrics_test], axis=0)

    # add threshold and maximum pnl to metrics
    metrics_train.loc["threshold", "Value"] = threshold_train
    metrics_valid.loc["threshold", "Value"] = threshold_valid
    metrics_test.loc["threshold", "Value"] = threshold_test
    metrics_train.loc["maximum_pnl", "Value"] = maximum_pnl_train
    metrics_valid.loc["maximum_pnl", "Value"] = maximum_pnl_valid
    metrics_test.loc["maximum_pnl", "Value"] = maximum_pnl_test

    # Log additional metrics for tracking
    for metric_name, row in metrics_train.iterrows():
        trial.set_user_attr(f"{metric_name}_train", row["Value"])
    for metric_name, row in metrics_valid.iterrows():
        trial.set_user_attr(f"{metric_name}_valid", row["Value"])
    for metric_name, row in metrics_test.iterrows():
        trial.set_user_attr(f"{metric_name}_test", row["Value"])

    # define the single metric to optimize on
    metric_to_optimize = metrics_test.loc[name_metric_to_optimize, "Value"]

    # return single metric to optimize on
    return metric_to_optimize
