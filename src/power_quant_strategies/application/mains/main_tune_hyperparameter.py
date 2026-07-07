import os
from pathlib import Path

import pandas as pd

from power_quant_strategies.application.settings.models import BaseSettings
from power_quant_strategies.utils.fit_model.fit_xgboost.tune_hyperparameter_xgboost import tune_hyperparameter_xgboost
from power_quant_strategies.utils.helper.setup_logging import setup_logging


def main_tune_hyperparameter(data: pd.DataFrame, settings: BaseSettings, save_path: str = str(Path.cwd())) -> dict:
    """
    Tune hyperparameter.

    :param data: DataFrame containing features and target.
    :param settings: BaseSettings object
    :param save_path: Directory for saving tuning results.
    :return: Dictionary containing the best hyperparameters.
    """
    # initialize logger
    setup_logging()

    # tune hyperparameter settings
    save_path_hyperparameter = os.path.join(save_path, "tune_hyperparameter")
    os.makedirs(name=save_path_hyperparameter, exist_ok=True)

    ## prepare data
    # ensure the index is datetimeindex
    data.index = pd.to_datetime(data.index)

    # shuffle the dataset before splitting to create random
    data_train_valid_test = data.sample(frac=1, random_state=42)

    # compute split indices
    end_train = int(len(data_train_valid_test) * settings.percentage_train_hyperparameter_tuning)
    end_valid = int(len(data_train_valid_test) * (settings.percentage_train_hyperparameter_tuning + settings.percentage_valid_hyperparameter_tuning))
    data_train = data_train_valid_test[:end_train].copy()
    data_valid = data_train_valid_test[end_train:end_valid].copy()
    data_test = data_train_valid_test[end_valid:].copy()

    # run optuna hyperparameter optimization
    study, best_params = tune_hyperparameter_xgboost(
        data_train=data_train,
        data_valid=data_valid,
        data_test=data_test,
        column_name_target=settings.column_name_target,
        n_estimators=settings.n_estimators,
        early_stopping_rounds=settings.early_stopping_rounds,
        verbose=settings.verbose,
        random_state=settings.random_state,
        direction=settings.direction,
        n_trials=settings.n_trials,
        max_run_time_per_model_fit=settings.max_run_time_per_model_fit,
        show_progress_bar=settings.show_progress_bar,
        save_path=save_path_hyperparameter,
        device=settings.device,
    )

    # convert the best hyperparameters into a standard dictionary
    hyperparameter = best_params["Value"].to_dict() if not isinstance(best_params, dict) else best_params["Value"]

    return hyperparameter
