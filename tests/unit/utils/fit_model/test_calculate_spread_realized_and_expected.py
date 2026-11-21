import numpy as np
import pandas as pd
import pytest

from power_quant_strategies.utils.fit_model.calculate_spread_realized_and_expected import (
    _fit_spread_expected,
    add_spread_realized_and_expected,
    calculate_spread_expected,
    calculate_spread_realized,
)


def _split(trading_data):
    data = trading_data.drop(columns="preds")
    return data.iloc[:60].copy(), data.iloc[60:80].copy(), data.iloc[80:].copy()


def test_calculate_spread_realized():
    data = pd.DataFrame({"target": [1, 0], "sample_weight": [3.0, 2.0]})
    assert calculate_spread_realized(data, "target").tolist() == [3.0, -2.0]


def test_calculate_spread_realized_missing_column():
    with pytest.raises(KeyError, match="sample_weight"):
        calculate_spread_realized(pd.DataFrame({"target": [1]}), "target")


def test_fit_spread_expected_learns_a_linear_relation():
    rng = np.random.default_rng(0)
    features = pd.DataFrame({"x": rng.normal(size=200), "mostly_nan": np.nan}, index=pd.date_range("2026-01-01", periods=200, freq="h"))
    features.iloc[0, 1] = 1.0
    target = 3 * features["x"] + rng.normal(scale=0.01, size=200)

    model, feature_mask = _fit_spread_expected(features, target)

    assert list(feature_mask) == ["x"]  # the NaN feature was dropped
    prediction = calculate_spread_expected(model, features, feature_mask)
    assert np.allclose(prediction, target, atol=0.1)


def test_calculate_spread_expected_is_nan_without_model_or_with_nans():
    features = pd.DataFrame({"x": [1.0, np.nan]})
    result = calculate_spread_expected(None, features, pd.Index(["x"]))
    assert result.isna().all()
    assert result.name == "spread_expected"


def test_add_spread_realized_and_expected(trading_data):
    data_train, data_valid, data_test = _split(trading_data)

    data_train, data_valid, data_test = add_spread_realized_and_expected(data_train, data_valid, data_test, column_name_target="target")

    for data in (data_train, data_valid, data_test):
        assert {"spread_realized", "spread_expected"} <= set(data.columns)
        assert data["spread_expected"].notna().all()
    assert (data_test["spread_realized"] == (2 * data_test["target"] - 1) * data_test["sample_weight"]).all()


def test_add_spread_skips_when_already_present(trading_data):
    data_train, data_valid, data_test = _split(trading_data)
    data_train["spread_realized"] = 0.0

    result = add_spread_realized_and_expected(data_train, data_valid, data_test, column_name_target="target")

    assert "spread_expected" not in result[0].columns
