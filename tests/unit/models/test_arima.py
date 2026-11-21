import numpy as np
import pandas as pd

from power_quant_strategies.models.arima import arima_model

PARAMETERS = {"order": (1, 0, 0), "seasonal_order": (0, 0, 0, 0), "enforce_stationarity": True, "enforce_invertibility": True}


def _ar1(n: int = 300) -> pd.Series:
    rng = np.random.default_rng(0)
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = 0.7 * values[i - 1] + rng.normal()
    return pd.Series(values, index=pd.date_range("2026-01-01", periods=n, freq="h"))


def test_fits_an_ar1_process():
    model = arima_model(endog=_ar1(), exog=pd.DataFrame(), parameters=PARAMETERS)
    result = model.fit()
    assert abs(result.params["ar.L1"] - 0.7) < 0.1


def test_empty_exog_means_no_exogenous_variables():
    model = arima_model(endog=_ar1(), exog=pd.DataFrame(), parameters=PARAMETERS)
    assert model.k_exog == 0


def test_uses_exogenous_variables():
    endog = _ar1()
    exog = pd.DataFrame({"x": np.random.default_rng(1).normal(size=len(endog))}, index=endog.index)
    model = arima_model(endog=endog, exog=exog, parameters=PARAMETERS)
    assert "x" in model.exog_names
