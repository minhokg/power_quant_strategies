import os
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

from power_quant_strategies.utils.helper.convert_to_snake_case import convert_to_snake_case
from power_quant_strategies.utils.helper.shorten_path import shorten_path


def check_autocorrelation(
    data: pd.DataFrame,
    save_path: str = os.getcwd(),
    acf_pacf_lags: Union[int, None] = None,
    lb_lags: Optional[Union[int, Sequence[int]]] = None,
    lb_boxpierce: bool = False,
    lb_model_df: int = 0,
    lb_period: int | None = None,
    lb_return_df: bool = True,
    lb_auto_lag: bool = True,
) -> pd.DataFrame:
    """
    Check each column of data for autocorrelation.

    :param data: input data with different variables in columns and time in rows
    :param save_path: defines the path to save the figures.
    :param acf_pacf_lags: number of lags to use for acf and pacf
    :param lb_lags: if lags is an integer then this is taken to be the largest lag that is included, the test result is
    reported for all smaller lag length. If lags is a list or array, then all lags are included up to the largest lag in
    the list
    :param lb_boxpierce: If True, also compute the Box-Pierce test.
    :param lb_model_df: Degrees of freedom used by the model (subtracted from the lags), e.g. p + q of an ARMA model.
    :param lb_period: Seasonal period of the series, used to choose the lags when lb_lags is None.
    :param lb_return_df: Return the Ljung-Box result as a DataFrame (needed for the metrics below).
    :param lb_auto_lag: Choose the lags automatically from the largest significant autocorrelation.
    :return: Ljung-Box and runs test statistics and p-values, one row per column of data.
    """
    # Create path to save results in
    if save_path:
        os.makedirs(name=save_path, exist_ok=True)
        os.makedirs(name=os.path.join(save_path, "01_explore_data"), exist_ok=True)

    # Initialize an empty dict to store metrics in
    metrics = {}

    for column in data.columns:
        # Plot acf and pacf
        # Autocorrelation function (acf):
        if acf_pacf_lags is None:
            n_obs = len(data[column].dropna())
            lags = int(np.sqrt(n_obs))
        else:
            lags = acf_pacf_lags
        plt.figure()
        plot_acf(data[column].dropna(), lags=lags, title=f"ACF for {column} (lags={lags})")
        plt.grid(True)
        plt.tight_layout()
        plt.gcf().set_facecolor("white")
        if save_path:
            plt.savefig(shorten_path(os.path.join(save_path, f"autocorrelation_acf_{convert_to_snake_case(column)}.png")), format="png")
        plt.close()
        plt.figure()
        plot_pacf(data[column].dropna(), lags=lags, title=f"PACF for {column} (lags={lags})")
        plt.grid(True)
        plt.tight_layout()
        plt.gcf().set_facecolor("white")
        if save_path:
            plt.savefig(shorten_path(os.path.join(save_path, f"autocorrelation_pacf_{convert_to_snake_case(column)}.png")), format="png")
        plt.close()

        # Run ljung-box test
        # The ljung-box test checks whether a time series exhibits autocorrelation at multiple lags simultaneously.
        # It is often applied to residuals from a model to verify if the remaining error terms are uncorrelated (white noise)
        # Null Hypothesis (H0):
        # The data has no significant autocorrelation (white noise).
        # Alternative Hypothesis (H1):
        # The data exhibits significant autocorrelation at one or more lags.
        # Interpretation:
        # If the p-value < 0.05, reject the null hypothesis -> significant autocorrelation exists.
        # If the p-value >= 0.05, fail to reject the null hypothesis -> no significant autocorrelation detected.
        ljung_box_metrics = acorr_ljungbox(
            x=data[column].dropna() - data[column].dropna().mean(),
            lags=lb_lags,
            boxpierce=lb_boxpierce,
            model_df=lb_model_df,
            period=lb_period,
            return_df=lb_return_df,
            auto_lag=lb_auto_lag,
        )

        # Run runs test
        # The Runs Test evaluates whether the sequence of values in a dataset is randon based on runs (consecutive
        # values above or below the median).
        # Null Hypothesis (H0): the sequence is random.
        # Alternative Hypothesis (H1): The sequence is not random (indicating potential patterns, trends, or autocorrelation).
        # Interpretation:
        # If the p-value < 0.05, reject the null hypothesis -> the sequence is not random.
        # If the p-value >= 0.05, fail to reject the null hypothesis -> the sequence appears random.
        runs_test_metrics = _runs_test(data=data[column].dropna())

        metrics[column] = {
            "ljung_box_statistic": ljung_box_metrics["lb_stat"].iloc[0],
            "ljung_box_pvalue": ljung_box_metrics["lb_pvalue"].iloc[0],
            "runs_statistic": runs_test_metrics["z-statistic"],
            "runs_pvalue": runs_test_metrics["p-value"],
        }

    # convert the metrics dictionary to a DataFrame
    metrics = pd.DataFrame(metrics).T

    # save to disk
    if save_path:
        metrics.to_parquet(path=os.path.join(save_path, "01_explore_data", "autocorrelation_metrics.parquet"), index=True)
        metrics.to_excel(excel_writer=os.path.join(save_path, "01_explore_data", "autocorrelation_metrics.xlsx"))

    # return metrics with test statistics, p-values, and other relevant results
    return metrics


def _runs_test(data: Sequence[float]) -> dict:
    """
    Perform the Runs Test (Wald-Wolfowitz Test) to check the randomness of a sequence.

    The Runs Test assesses whether the number of runs (consecutive sequences of values above or below the median)
    in the input data deviates significantly from randomness.
    Non-randomness can suggest autocorrelation or trends in the data.

    :param data: A sequence of numerical values (e.g., list, numpy array).
    :return: A dict containing: The Z-statistics for the test. The p-value, used to assess the null hypothesis of randomness.
    """
    # make sure there are at least 2 data points
    if len(data) < 2:
        raise ValueError("Input data must contain at least 2 elements.")

    # calculate the median of the data
    median = np.median(data)

    # generate a sequence of signs: 1 if value > median, -1 otherwise
    signs = np.where(data > median, 1, -1)

    # count the number of runs
    runs = 1 + np.sum(np.diff(signs) != 0)

    # calculate n1 (number of positive signs) and n2 (number of negative signs)
    n1 = np.sum(signs == 1)
    n2 = np.sum(signs == -1)

    # calculate the expected number of runs under randomness
    expected_runs = ((2 * n1 * n2) / (n1 + n2)) + 1

    # calculate the variance of the runs
    variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))

    # calculate the Z-statistic
    z_statistic = (runs - expected_runs) / np.sqrt(variance_runs)

    # calculate the two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(z_statistic)))

    # create dict
    metrics = {"z-statistic": z_statistic, "p-value": p_value}

    # return dict containing: The Z-statistic for the test. The p-value, used to assess the null hypothesis of
    # randomness
    return metrics
