import logging

import numpy as np
import pandas as pd


def define_target_and_sample_weights(data: pd.DataFrame, column_name_entry: str, column_name_exit: str, column_name_target: str = "target") -> pd.DataFrame:
    """
    Define target and sample weights.

    :param data: Original data with columns listed
    :param column_name_entry: Defines the column name in which to find prices to enter the position.
    :param column_name_exit: Defines the column name in which to find prices to exit the position.
    :param column_name_target: Defines the column in which the target will be stored.
    :return: Data enriched by target and sample weights.
    """
    # give user feedback
    logging.info("define target and sample weights")

    assert column_name_entry in data.columns.tolist(), f"{column_name_entry} not in data. Available columns: {data.columns}"
    assert column_name_exit in data.columns.tolist(), f"{column_name_exit} not in data. Available columns: {data.columns}"

    target_relevant_cols = [column_name_entry, column_name_exit]

    data_help = data.copy()

    data_help["sample_weight"] = np.abs(data_help[column_name_exit] - data_help[column_name_entry])

    data_help[column_name_target] = data_help[column_name_exit] - data_help[column_name_entry] > 0

    data_help = data_help.drop(labels=target_relevant_cols, axis=1)

    return data_help
