import os

import matplotlib.pyplot as plt
import pandas as pd

from power_quant_strategies.utils.helper.shorten_path import shorten_path


def evaluate_trading_plots(data: pd.DataFrame, save_path: str = os.getcwd()) -> None:
    """
    Evaluate trading plots.

    :param data: Original data
    :param save_path: Directory to save the plot
    :return: None
    """
    # calculate spread, trading_direction, and bet_size
    data_copy = data.copy()
    data_copy = data_copy.sort_index()
    data_copy["bet_size"] = (data_copy["preds"] - data_copy["threshold_train_valid"]) / (data_copy["preds"].max() - data_copy["threshold_train_valid"]) * (
        data_copy["preds"] >= data_copy["threshold_train_valid"]
    ) + (data_copy["preds"] - data_copy["threshold_train_valid"]) / (data_copy["preds"].min() - data_copy["threshold_train_valid"]) * (data_copy["preds"] <= data_copy["threshold_train_valid"])

    # calculate trading direction
    data_copy.loc[data_copy["preds"] >= data_copy["threshold_train_valid"], "trading_direction"] = 1
    data_copy.loc[data_copy["preds"] < data_copy["threshold_train_valid"], "trading_direction"] = -1

    # set spread_always initially to realized spread
    data_copy["spread_always"] = data_copy["spread_realized"]

    # calculate pnl
    data_copy["pnl_long"] = data_copy["spread_realized"] * (data_copy["trading_direction"] == 1) * data_copy["bet_size"] / data_copy["bet_size"].mean()
    data_copy["pnl_short"] = data_copy["spread_realized"] * (data_copy["trading_direction"] == -1) * data_copy["bet_size"] * (-1) / data_copy["bet_size"].mean()
    data_copy["pnl"] = data_copy["pnl_long"] + data_copy["pnl_short"]

    # calculate cumulative pnl
    data_copy["pnl_long_cumulative"] = data_copy["pnl_long"].cumsum()
    data_copy["pnl_short_cumulative"] = data_copy["pnl_short"].cumsum()
    data_copy["pnl_cumulative"] = data_copy["pnl"].cumsum()
    data_copy["pnl_always_long_cumulative"] = data_copy["spread_always"].cumsum()
    data_copy["pnl_always_short_cumulative"] = data_copy["spread_always"].cumsum() * (-1)

    os.makedirs(name=os.path.join(save_path, "03_evaluate_model"), exist_ok=True)
    data_copy.to_parquet(os.path.join(save_path, "03_evaluate_model", "data_test_all_with_pnl.parquet"))

    # plot - whole history
    plt.figure()
    plt.plot(data_copy[["pnl_cumulative"]], label="Pnl Cumulative", color="black")
    plt.plot(data_copy[["pnl_long_cumulative"]], label="Pnl Long Cumulative", color="red")
    plt.plot(data_copy[["pnl_short_cumulative"]], label="Pnl Short Cumulative", color="blue")
    plt.plot(data_copy[["pnl_always_long_cumulative"]], label="Pnl Always Long Cumulative", color="red", linestyle="--")
    plt.plot(data_copy[["pnl_always_short_cumulative"]], label="Pnl Always Short Cumulative", color="blue", linestyle="--")
    plt.title("Cumulative PnL vs Time")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.gcf().set_facecolor("white")
    plt.savefig(shorten_path(os.path.join(save_path, "03_evaluate_model", "walk_forward_cumulative_pnl.png")))
    plt.close()

    # plot - last year
    data_last_year = data_copy[data_copy.index >= (data_copy.index.max() - pd.DateOffset(years=1))]
    plt.figure()
    plt.plot(data_last_year[["pnl_cumulative"]] - data_last_year[["pnl_cumulative"]].iloc[0], label="Pnl Cumulative", color="black")
    plt.plot(data_last_year[["pnl_long_cumulative"]] - data_last_year[["pnl_long_cumulative"]].iloc[0], label="Pnl Long Cumulative", color="red")
    plt.plot(data_last_year[["pnl_short_cumulative"]] - data_last_year[["pnl_short_cumulative"]].iloc[0], label="Pnl Short Cumulative", color="blue")
    plt.plot(data_last_year[["pnl_always_long_cumulative"]] - data_last_year[["pnl_always_long_cumulative"]].iloc[0], label="Pnl Always Long Cumulative", color="red", linestyle="--")
    plt.plot(data_last_year[["pnl_always_short_cumulative"]] - data_last_year[["pnl_always_short_cumulative"]].iloc[0], label="Pnl Always Short Cumulative", color="blue", linestyle="--")
    plt.title("Cumulative PnL vs Time")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.gcf().set_facecolor("white")
    plt.savefig(shorten_path(os.path.join(save_path, "03_evaluate_model", "walk_forward_cumulative_pnl_last_year.png")))
    plt.close()

    # return None
    return None
