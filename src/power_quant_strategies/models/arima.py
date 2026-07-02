import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def arima_model(
    endog: pd.Series,
    exog: pd.DataFrame,
    parameters: dict = None,
) -> ARIMA:
    """

    Fit an ARIMA model for time-series forecasting.

    :param endog: Time series data.
    :param exog: Exogen data.
    :param parameters: Dictionary of model parameters.
    :return: ARIMA model.
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
