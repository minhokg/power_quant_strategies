import logging
import os
from typing import Union

import numpy as np
import pandas as pd


def evaluate_trading_metrics(
    data: pd.DataFrame,
    column_name_target: str,
    split: str,
    threshold: float,
    save_path: Union[str, None] = os.getcwd(),
) -> pd.DataFrame:
    """
    Evaluate trading metrics.

    :param data: Original data with columns 'sample_weight', 'preds' and column_name_target.
    :param column_name_target: Defines the target column name.
    :param split: Defines the cross validation split.
    :param threshold: Sets the threshold above with to go long and below which to go short.
    :param save_path: Defines where to save the results.
    :return: trading metrics with metric names as the index and their values in a column named "Value"
    """
    # give user feedback
    logging.info("evaluate trading metrics")

    # create a copy of the data to not alter the original
    data_copy = data.copy()
    data_copy = data_copy.sort_index()

    # calculate scaling factors
    n_obs_per_day = 24
    pnl_scaling_factor = (365.25 * n_obs_per_day) / data_copy.shape[0]
    sharpe_scaling_factor = np.sqrt(365.25)  # because Citadel-wide sharpe is defined over the daily pnl

    # calculate helper variables
    data_copy["spread"] = (2 * data_copy[column_name_target] - 1) * data_copy["sample_weight"]
    data_copy["trading_direction"] = 2 * (data_copy["preds"] >= threshold) - 1
    data_copy["bet_size"] = (data_copy["preds"] - threshold) / (data_copy["preds"].max() - threshold) * (data_copy["preds"] >= threshold) + (data_copy["preds"] - threshold) / (
        data_copy["preds"].min() - threshold
    ) * (data_copy["preds"] <= threshold)
    data_copy["pnl_long"] = data_copy["spread"] * (data_copy["trading_direction"] == 1) * data_copy["bet_size"] / data_copy["bet_size"].mean()
    data_copy["pnl_short"] = data_copy["spread"] * (data_copy["trading_direction"] == -1) * data_copy["bet_size"] * (-1) / data_copy["bet_size"].mean()
    data_copy["pnl"] = data_copy["pnl_long"] + data_copy["pnl_short"]
    # we scale the mean pnl per day with the number of observation per day
    # this is because this function might also be called with data that is incomplete (think of training set with shuffle = True)
    # due to shuffling data is missing completely at random (MCAR)
    # in that case the mean is an unbiased estimator for the missing data
    daily_pnl = data_copy.groupby(data_copy.index.normalize())["pnl"].mean() * n_obs_per_day

    # calculate trading metrics
    metrics = {}
    metrics["n_time_slices"] = data_copy.shape[0]
    metrics["pnl_long"] = data_copy["pnl_long"].sum()
    metrics["pnl_short"] = data_copy["pnl_short"].sum()
    metrics["pnl"] = data_copy["pnl"].sum()
    metrics["pnl_yearly"] = metrics["pnl"] * pnl_scaling_factor
    metrics["mean_pnl"] = data_copy["pnl"].mean()
    metrics["std_pnl"] = data_copy["pnl"].std()
    denominator_profit_factor = data_copy.loc[data_copy["pnl"] < 0, "pnl"].abs().sum()
    if denominator_profit_factor == 0:
        logging.warning("Profit factor is infinite because there are no losing trades.")
        metrics["profit_factor"] = np.inf
    else:
        metrics["profit_factor"] = data_copy.loc[data_copy["pnl"] > 0, "pnl"].sum() / denominator_profit_factor
    metrics["win_rate"] = (data_copy["pnl"] > 0).mean()
    # For max draw down we first convert incremental PnL values to cumulative PnL via cumsum().
    # cummax() then tracks the running peak of the cumulative PnL at every point in time.
    # The drawdown at each step is defined as: (historical peak PnL) – (current cumulative PnL).
    # Taking .max() over this difference gives the maximum draw down of the PnL series.
    metrics["max_draw_down"] = (data_copy["pnl"].cumsum().cummax() - data_copy["pnl"].cumsum()).max()
    metrics["sharpe_ratio"] = daily_pnl.mean() / daily_pnl.std() * sharpe_scaling_factor

    # convert metrics to DataFrame with index as metric names and column as "Value"
    metrics = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])

    # Sort DataFrame alphabetically by metric name
    metrics.sort_index(inplace=True)

    # save metrics to disk
    if save_path is not None:
        metrics.to_parquet(path=os.path.join(save_path, "03_evaluate_model", f"trading_metrics_{split}.parquet"), index=True)
        metrics.to_excel(excel_writer=os.path.join(save_path, "03_evaluate_model", f"trading_metrics_{split}.xlsx"))

    # return trading metrics with metric names as the index and their values in a column named "Value"
    return metrics
