import os
from pathlib import Path

import pandas as pd

from power_quant_strategies.application.settings.models import BaseSettings
from power_quant_strategies.utils.evaluate_model.evaluate_trading_plots import evaluate_trading_plots
from power_quant_strategies.utils.fit_model.fit_xgboost.fit_xgboost_walk_forward import fit_xgboost_walk_forward


def main_train_walk_forward(data: pd.DataFrame, hyperparameter: dict, save_path: str = str(Path.cwd())) -> None:
    """
    Train walk forward.

    :param data: DataFrame used for training walk forward methodology
    :param hyperparameter: Hyperparameter used for training
    :param save_path: Save path for saving results
    :return: None
    """
    # create necessary folders
    save_path_walk_forward = os.path.join(save_path, "train_walk_forward")
    os.makedirs(name=save_path_walk_forward, exist_ok=True)
    os.makedirs(name=os.path.join(save_path_walk_forward, "01_explore_data"), exist_ok=True)
    os.makedirs(name=os.path.join(save_path_walk_forward, "02_fit_model"), exist_ok=True)
    os.makedirs(name=os.path.join(save_path_walk_forward, "03_evaluate_model"), exist_ok=True)

    # train xgboost with walk forward
    (
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
    ) = fit_xgboost_walk_forward(
        data=data,
        column_name_target=BaseSettings.column_name_target,
        hyperparameter=hyperparameter,
        shuffle=BaseSettings.shuffle,
        verbose=BaseSettings.verbose,
        save_path=save_path_walk_forward,
    )

    # create backtesting plots
    evaluate_trading_plots(data=data_test_all, save_path=save_path_walk_forward)
