import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def arima_model(
    endog: pd.Series,
    exog: pd.DataFrame,
    parameters: dict = None,
) -> ARIMA:
    """

    Fit an ARIMA model for time-series forecasting.

    Args:
        endog (pd.Series): The time-series (y training) dataset to fit for forecasting.
        exog (pd.DataFrame): The time-series of X exogenous regressors.
        parameters (dict): Dictionary of parameters to configure the ARIMA model.

    Returns:
        ARIMA: Returns an instantiated (not yet fitted) ARIMA model

    """
    # Check else modeling error
    if exog.empty:
        exog = None

    model = ARIMA(
        endog=endog,
        exog=exog,
        order=parameters["order"],
        seasonal_order=parameters["seasonal_order"],
        enforce_stationarity=parameters["enforce_stationarity"],
        enforce_invertibility=parameters["enforce_invertibility"],
    )

    return model
