import numpy as np
import pandas as pd
import pytest

from power_quant_strategies.utils.explore_data.find_best_parametric_distribution_fit import _calculate_aic_bic, find_best_parametric_distribution_fit


def test_calculate_aic_bic():
    aic, bic = _calculate_aic_bic(log_likelihood=-100.0, num_params=2, num_data=50)
    assert aic == pytest.approx(204.0)
    assert bic == pytest.approx(np.log(50) * 2 + 200.0)


def test_normal_data_fits_normal_best(tmp_path):
    data = pd.DataFrame({"price": np.random.default_rng(0).normal(loc=0, scale=1, size=2000)})

    results, qq_plots = find_best_parametric_distribution_fit(data, save_path=str(tmp_path))

    assert results.iloc[0]["Distribution"] == "norm"
    assert set(results["Distribution"]) == {"norm", "expon", "laplace"}  # some values <= 0: no lognorm
    assert (results["Column"] == "price").all()
    assert len(qq_plots) == 3
    assert (tmp_path / "01_explore_data" / "best_parametric_distribution_fit_statistics_price.parquet").exists()
    assert (tmp_path / "01_explore_data" / "best_parametric_distribution_fit_norm.png").exists()


def test_positive_data_also_tries_lognormal(tmp_path):
    data = pd.DataFrame({"x": np.random.default_rng(0).lognormal(size=500)})
    results, _ = find_best_parametric_distribution_fit(data, save_path=str(tmp_path))
    assert "lognorm" in set(results["Distribution"])
    assert results.iloc[0]["Distribution"] == "lognorm"


def test_ignores_non_finite_values(tmp_path):
    values = np.random.default_rng(0).normal(size=200)
    values[:5] = [np.nan, np.inf, -np.inf, np.nan, np.nan]
    results, _ = find_best_parametric_distribution_fit(pd.DataFrame({"x": values}), save_path=str(tmp_path))
    assert np.isfinite(results["AIC"]).all()


def test_needs_exactly_one_column(tmp_path):
    with pytest.raises(ValueError):
        find_best_parametric_distribution_fit(pd.DataFrame({"a": [1.0], "b": [2.0]}), save_path=str(tmp_path))
