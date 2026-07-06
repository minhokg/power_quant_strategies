import logging
import os
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

from power_quant_strategies.utils.helper.shorten_path import shorten_path


def find_optimal_threshold_and_maximum_pnl(
    data: pd.DataFrame,
    column_name_target: str,
    split: str,
    save_path: Union[str, None] = os.getcwd(),
    postfix: str = "full",
) -> tuple:
    """
    Find optimal threshold and maximum pnl.

    :param data: Original data
    :param column_name_target: Defines the target column name
    :param split: Defines the split (train, valid, test)
    :param save_path: Directory to save the optimal thresholds, maximum pnl and the plot
    :param postfix: Postfix for the plot file name
    :return: threshold and maximum_pnl
    """
    # give user feedback
    logging.info("find optimal threshold and maximum pnl")

    # create copy of data to not mess with the original data
    data_copy = data.copy()

    # calculate the spread
    data_copy["spread"] = data_copy["sample_weight"] * (2 * data_copy[column_name_target] - 1)

    # standardize spread by days
    data_copy["spread"] = data_copy["spread"] / pd.Series(data_copy.index.date).nunique()

    # sort by preds in descending order
    data_copy = data_copy.sort_values(by="preds", ascending=False)

    # calculate cumulative pnl long
    data_copy["cumsum_pnl_long"] = data_copy["spread"].cumsum()

    # sort by preds in ascending order
    data_copy = data_copy.sort_values(by="preds", ascending=True)

    # calculate cumulative pnl short
    data_copy["cumsum_pnl_short"] = data_copy["spread"].cumsum() * (-1)

    # sum pnl long and pnl short for total pnl
    data_copy["cumsum_pnl"] = data_copy["cumsum_pnl_long"] + data_copy["cumsum_pnl_short"]

    # determine maximum pnl
    maximum_pnl = data_copy["cumsum_pnl"].max()
    # determine optimal threshold
    threshold = data_copy.loc[data_copy["cumsum_pnl"].idxmax(), "preds"]

    # determine benchmarks
    long_only = data_copy["spread"].sum()
    short_only = data_copy["spread"].sum() * (-1)

    # save threshold and maximum pnl to disk
    if save_path is not None:
        os.makedirs(name=os.path.join(save_path, "02_fit_model"), exist_ok=True)
        threshold_and_maximum_pnl = {"threshold": threshold, "maximum_pnl_per_day": maximum_pnl}
        threshold_and_maximum_pnl = pd.DataFrame.from_dict(threshold_and_maximum_pnl, orient="index", columns=["Value"])
        threshold_and_maximum_pnl.to_parquet(path=os.path.join(save_path, "02_fit_model", f"threshold_and_maximum_pnl_{split}.parquet"), index=True)
        threshold_and_maximum_pnl.to_excel(excel_writer=os.path.join(save_path, "02_fit_model", f"threshold_and_maximum_pnl_{split}.xlsx"))

    # plot cumsum pnl
    plt.figure()
    plt.plot(data_copy["preds"], data_copy["cumsum_pnl"], color="black", label="Cumulative Pnl per Day")
    plt.axvline(x=threshold, linestyle="--", color="yellow", label="Optimal Threshold")
    plt.axhline(y=long_only, linestyle="--", color="blue", label="Long Only")
    plt.axhline(y=short_only, linestyle="--", color="red", label="Short Only")
    plt.title(f"Cumulative PnL per Day - split = {split}")
    plt.xlabel(xlabel="Preds")
    plt.ylabel(ylabel="Cumsum Pnl per Day")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.gcf().set_facecolor("white")
    if save_path is not None:
        plt.savefig(shorten_path(os.path.join(save_path, "02_fit_model", f"cumulative_pnl_per_day_{split}_{postfix}.png")))
    plt.close()

    # return threshold and maximum_pnl
    return threshold, maximum_pnl
