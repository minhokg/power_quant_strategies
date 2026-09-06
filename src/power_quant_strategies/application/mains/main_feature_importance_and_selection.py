import os

import pandas as pd

from power_quant_strategies.application.settings.models import BaseSettings
from power_quant_strategies.utils.explore_data.calculate_feature_importance import calculate_feature_importance
from power_quant_strategies.utils.helper.setup_logging import setup_logging
from power_quant_strategies.utils.preprocess_data.select_features import select_features


def main_feature_importance_and_selection(data: pd.DataFrame, settings: BaseSettings) -> list:
    """
    Run feature selection algorithm.

    :param data: DataFrame containing features and target.
    :param settings: BaseSettings object.
    :return: Selected feature list.
    """
    # initialize logger
    setup_logging()

    # set save_path
    save_path = os.path.join(settings.save_path, "feature_selection")
    os.makedirs(name=save_path, exist_ok=True)

    # calculate feature importance
    feature_importance = calculate_feature_importance(data=data, save_path=save_path)

    # select feature using boruta-like style
    selected_features = select_features(
        feature_importance=feature_importance,
        feature_selection_strategy=settings.feature_selection_strategy,
        save_path=save_path,
    )

    return selected_features
