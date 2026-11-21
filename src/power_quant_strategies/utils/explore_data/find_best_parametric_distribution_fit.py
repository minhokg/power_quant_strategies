import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import expon, kstest, laplace, lognorm, norm, probplot

from power_quant_strategies.utils.helper.shorten_path import shorten_path


def _calculate_aic_bic(log_likelihood: float, num_params: int, num_data: int) -> tuple[float, float]:
    """
    Calculate the Akaike and Bayesian information criteria.

    :param log_likelihood: Log-likelihood of the fitted distribution.
    :param num_params: Number of fitted parameters.
    :param num_data: Number of data points the distribution was fitted on.
    :return: AIC and BIC
    """
    aic = 2 * num_params - 2 * log_likelihood
    bic = np.log(num_data) * num_params - 2 * log_likelihood
    return aic, bic


def find_best_parametric_distribution_fit(data: pd.DataFrame, save_path: str = os.getcwd()) -> tuple:
    """
    Fits the given data to parametric distributions and evaluates each fit.

    Normal, exponential and Laplace are always tried; log-normal only if all values are positive. Non-finite values
    are ignored. Results are sorted by AIC (best first).

    :param data: DataFrame with exactly one column to be evaluated.
    :param save_path: Directory to save the results and plots (in its 01_explore_data folder)
    :raises ValueError: If data does not have exactly one column.
    :return: tuple with distribution evaluation metrics and a list of qq plots
    """
    if data.shape[1] != 1:
        raise ValueError("data must contain exactly one column")

    # define list of allowed distributions
    distributions = [norm, expon, laplace]

    x = data.squeeze().to_numpy()

    # check and remove non-finite values
    if not np.all(np.isfinite(x)):
        x = x[np.isfinite(x)]

    os.makedirs(os.path.join(save_path, "01_explore_data"), exist_ok=True)

    if np.all(x > 0):
        distributions.append(lognorm)

    # evaluate each distribution
    results = []
    qq_plots = []
    for dist in distributions:
        try:
            # fit distribution to data
            params = dist.fit(x)

            # log-likelihood
            log_likelihood = np.sum(dist.logpdf(x, *params))

            # number of parameters
            num_params = len(params)

            # AIC and BIC
            aic, bic = _calculate_aic_bic(log_likelihood=log_likelihood, num_params=num_params, num_data=len(x))

            # kolmogorov-smirnov test
            ks_test, ks_p = kstest(rvs=x, cdf=dist.name, args=params)

            # store results
            results.append(
                {
                    "Distribution": dist.name,
                    "Log-Likelihood": log_likelihood,
                    "AIC": aic,
                    "BIC": bic,
                    "KS Statistic": ks_test,
                    "KS P-Value": ks_p,
                    "Parameters": params,
                }
            )

            # generate a qq plot for this distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            probplot(x, dist=dist, sparams=params, plot=ax)
            ax.set_title(f"Q-Q Plot: Data vs. {dist.name}")
            ax.grid()
            qq_plots.append(fig)
            plt.grid(True)
            plt.tight_layout()
            plt.gcf().set_facecolor("white")
            plt.savefig(shorten_path(os.path.join(save_path, "01_explore_data", f"best_parametric_distribution_fit_{dist.name}.png")))

        except ValueError as error:
            logging.error(f"Failed to fit {dist.name}: {error}")

    # convert results to DataFrame
    results = pd.DataFrame(results).sort_values(by="AIC")

    # add column name
    column_name = data.columns[0]
    results.insert(0, "Column", column_name)

    # save to disk
    results.to_parquet(
        path=os.path.join(
            save_path,
            "01_explore_data",
            f"best_parametric_distribution_fit_statistics_{column_name}.parquet",
        ),
        index=True,
    )
    results.to_excel(
        excel_writer=os.path.join(
            save_path,
            "01_explore_data",
            f"best_parametric_distribution_fit_statistics_{column_name}.xlsx",
        )
    )

    # return tuple with distribution evaluation metrics and a list of qq plots
    return results, qq_plots
