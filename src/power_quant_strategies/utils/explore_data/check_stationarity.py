import os
from typing import Literal, Union

import pandas as pd
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.stattools import adfuller, kpss


def check_stationarity(
    data: pd.DataFrame,
    adf_maxlag: int | None = None,
    adf_regression: Literal["c", "ct", "ctt", "n"] = "c",
    adf_autolag: Literal["AIC", "BIC", "t-stat", None] = "AIC",
    adf_store: bool = False,
    kpss_regression: Literal["c", "ct"] = "c",
    kpss_nlags: Union[str, int] = "auto",
    pp_lags: Union[int, None] = None,
    pp_trend: Literal["n", "c", "ct"] = "c",
    pp_test_type: Literal["tau", "rho"] = "tau",
    save_path: str | None = os.getcwd(),
) -> pd.DataFrame:
    """
    Check stationarity of data with Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test.

    The tests are run on each column individually. How to interpret the results? If ADF p-value is below a certain
    significance level alpha or the ADF statistic is below ADF Critical Value, then the time series is stationarity. For
    the KPSS it is the other way around. So if KPSS p-value is above alpha or KPSS Statistic is above KPSS Critical Value,
    then the time series is stationarity.

    :param data: Containing the data to be checked for stationarity. Each column is assumed to be a time series. The
    rows are the time axis sorted by ascending time
    :param adf_maxlag: Maximum lag which is included in test. None uses 12*(nobs/100)^{1/4}.
    :param adf_regression: Constant and trend order to include in regression. "c": constant only (default). "ct": constant
    and trend. "ctt": constant, and linear and quadratic trend. "n": no constant, no trend.
    :param adf_autolag: Method to use when automatically determining the lag length among the values 0, 1, ..., maxloag.
    :param adf_store: If True, then a result instance is returned additionally to the adf statistic. Default is False
    :param kpss_regression: The null hypothesis for the KPSS test. "c": The data is stationary around a constant (default).
    "ct": The data is stationary around a trend.
    :param kpss_nlags: Indicates the number of lags to be used. If “auto” (default), lags is calculated using the
    data-dependent method of Hobijn et al. (1998). See also Andrews (1991), Newey & West (1994), and Schwert (1989).
    If set to “legacy”, uses int(12 * (n / 100)**(1 / 4)) , as outlined in Schwert (1989).
    :param pp_lags: The number of lags to use in the Newey-West estimator of the long-run covariance. If omitted or
    None, the lag length is set automatically to 12 * (nobs/100) ** (1/4).
    :param pp_trend: The trend component to include in the test. ”n” - No trend components. ”c” - Include a constant
    (Default). ”ct” - Include a constant and linear time trend.
    :param pp_test_type: The test to use when computing the test statistic. “tau” is based on the t-stat and “rho” uses
    a test based on nobs times the re-centered regression coefficient.
    :param save_path: Directory to save the metrics (in its 01_explore_data folder), or None to skip saving
    :return: metrics from adf and kpss test
    """
    metrics = {}
    for column in data.columns:
        # run Augmented Dickey-Fuller (ADF) test
        # make sure to drop na as adf can't handle them
        # The ADF test checks whether a time series is stationary or contains a unit root. It is a one-sided test where
        # the null hypothesis states that the series has a unit root (non-stationary).
        # Interpretation:
        # p-value < 0.05: Reject the null hypothesis; the series is stationary.
        # p-value >= 0.05: Fail to reject the null hypothesis; the series is likely non-stationary.
        adf_metrics = adfuller(x=data[column].dropna(), maxlag=adf_maxlag, regression=adf_regression, autolag=adf_autolag, store=adf_store)
        if adf_metrics[1] < 0.05:
            adf_verdict = "stationary"
        else:
            adf_verdict = "likely non-stationary"

        # run Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
        # make sure to drop na as kpss can't handle them
        # The KPSS test assesses whether a time series is stationary around a deterministic trend or level. The null
        # hypothesis assumes the series is stationary.
        # Interpretation:
        # p-value < 0.05: Reject the null hypothesis; the series is non-stationary.
        # p-value >= 0.05: Fail to reject the null hypothesis; the series is likely stationary.
        kpss_metrics = kpss(x=data[column].dropna(), regression=kpss_regression, nlags=kpss_nlags)
        if kpss_metrics[1] < 0.05:
            kpss_verdict = "non-stationary"
        else:
            kpss_verdict = "likely stationary"

        # run Phillips Perron (PP) test
        # make sure to drop na as pp can't handle them
        # The PP test is similar to the ADF test but it is more robust to heteroskedasticity and serial correlation in the
        # error terms. The null hypothesis states that the series has a unit root (non-stationary).
        # Interpretation:
        # p-value < 0.05: Reject the null hypothesis; the series is stationary.
        # p-value >= 0.05: Fail to reject the null hypothesis; the series is likely non-stationary.
        pp_metrics = PhillipsPerron(y=data[column].dropna(), lags=pp_lags, trend=pp_trend, test_type=pp_test_type)
        if pp_metrics.pvalue < 0.05:
            pp_verdict = "stationary"
        else:
            pp_verdict = "likely non-stationary"

        # save metrics
        metrics[column] = {
            "adf_statistic": adf_metrics[0],
            "adf_pvalue": adf_metrics[1],
            "adf_lags_used": adf_metrics[2],
            "adf_number_of_observations": adf_metrics[3],
            "adf_critical_value_1p": adf_metrics[4]["1%"],
            "adf_critical_value_5p": adf_metrics[4]["5%"],
            "adf_critical_value_10p": adf_metrics[4]["10%"],
            "adf_maximized_information_criterion": adf_metrics[5],
            "adf_verdict": adf_verdict,
            "kpss_statistic": kpss_metrics[0],
            "kpss_pvalue": kpss_metrics[1],
            "kpss_lags_used": kpss_metrics[2],
            "kpss_critical_value_1p": kpss_metrics[3]["1%"],
            "kpss_critical_value_2.5p": kpss_metrics[3]["2.5%"],
            "kpss_critical_value_5p": kpss_metrics[3]["5%"],
            "kpss_critical_value_10p": kpss_metrics[3]["10%"],
            "kpss_verdict": kpss_verdict,
            "pp_statistic": pp_metrics.stat,
            "pp_pvalue": pp_metrics.pvalue,
            "pp_lags_used": pp_metrics.lags,
            "pp_number_of_observations": pp_metrics.nobs,
            "pp_critical_value_1p": pp_metrics.critical_values["1%"],
            "pp_critical_value_5p": pp_metrics.critical_values["5%"],
            "pp_critical_value_10p": pp_metrics.critical_values["10%"],
            "pp_verdict": pp_verdict,
        }

    # convert the metrics dictionary to a DataFrame
    metrics = pd.DataFrame(metrics).T

    # save to disk
    if save_path is not None:
        os.makedirs(os.path.join(save_path, "01_explore_data"), exist_ok=True)
        metrics.to_parquet(path=os.path.join(save_path, "01_explore_data", "stationarity_metrics.parquet"), index=True)
        metrics.to_excel(excel_writer=os.path.join(save_path, "01_explore_data", "stationarity_metrics.xlsx"))

    # return metrics from adf and kpss test
    return metrics
