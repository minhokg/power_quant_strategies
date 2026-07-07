import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

from power_quant_strategies.utils.helper.shorten_path import shorten_path


def check_model_score_vs_spread_dependency(
    data: pd.DataFrame,
    column_name_target: str,
    split: str,
    save_path: str = os.getcwd(),
) -> None:
    """
    Check model score vs spread dependency.

    The assumption is that we see an upward sloped line and that the data points scatter around the line.
    :param data: Original data with columns defined by column_name_target and sample_weight
    :param column_name_target: Defines the name of the target column
    :param split: Defines which data split (train, valid, test) was used for the plot
    :param save_path: Path to save the figure
    :return: None
    """
    # give user feedback
    logging.info("check model score vs spread dependency")

    # create copy of data to not interfere with the original data
    data_copy = data.copy()

    # calculate spread from target and sample weight
    data_copy["spread"] = (2 * data_copy[column_name_target].values - 1) * data_copy["sample_weight"]

    # fit linear regression on preds vs spread
    slope, intercept, _, _, _ = linregress(x=data_copy["preds"], y=data_copy["spread"])

    # plot
    plt.figure()
    plt.scatter(x=data_copy["preds"], y=data_copy["spread"], label="Data points")
    plt.plot(data_copy["preds"], slope * data_copy["preds"] + intercept, color="red", label="Regression line")
    plt.yscale(value="symlog", linthresh=max(abs(slope * data_copy["preds"] + intercept)))
    plt.xlabel(xlabel="Preds")
    plt.ylabel(ylabel="Spread")
    plt.title(label=f"Spread vs Preds - split = {split}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.gcf().set_facecolor("white")
    if save_path is not None:
        plt.savefig(
            shorten_path(
                os.path.join(
                    save_path,
                    "03_evaluate_model",
                    f"model_score_vs_spread_{split}.png",
                )
            )
        )
    plt.close()

    return None
