import numpy as np
import pandas as pd
import pytest

from power_quant_strategies.utils.explore_data.check_autocorrelation import _runs_test, check_autocorrelation


def test_runs_test_alternating_sequence_is_not_random():
    result = _runs_test(np.array([1.0, -1.0] * 50))
    assert result["z-statistic"] > 0  # far more runs than expected
    assert result["p-value"] < 0.05


def test_runs_test_two_blocks_are_not_random():
    result = _runs_test(np.array([-1.0] * 50 + [1.0] * 50))
    assert result["z-statistic"] < 0  # far fewer runs than expected
    assert result["p-value"] < 0.05


def test_runs_test_random_sequence_is_random():
    result = _runs_test(np.random.default_rng(0).normal(size=1000))
    assert result["p-value"] > 0.05


def test_runs_test_needs_two_values():
    with pytest.raises(ValueError):
        _runs_test(np.array([1.0]))


def test_check_autocorrelation_finds_autocorrelated_series(tmp_path):
    rng = np.random.default_rng(0)
    ar = np.zeros(400)
    for i in range(1, 400):
        ar[i] = 0.9 * ar[i - 1] + rng.normal()
    data = pd.DataFrame({"White Noise": rng.normal(size=400), "AR(1)": ar})

    metrics = check_autocorrelation(data, save_path=str(tmp_path), lb_lags=10, lb_auto_lag=False)

    assert list(metrics.columns) == ["ljung_box_statistic", "ljung_box_pvalue", "runs_statistic", "runs_pvalue"]
    assert metrics.loc["AR(1)", "ljung_box_pvalue"] < 0.05
    assert metrics.loc["AR(1)", "runs_pvalue"] < 0.05
    assert metrics.loc["White Noise", "ljung_box_pvalue"] > 0.05

    # plots use snake_case file names, metrics go to 01_explore_data
    assert (tmp_path / "autocorrelation_acf_ar_1.png").exists()
    assert (tmp_path / "autocorrelation_pacf_white_noise.png").exists()
    assert (tmp_path / "01_explore_data" / "autocorrelation_metrics.parquet").exists()


def test_check_autocorrelation_without_saving(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = pd.DataFrame({"x": np.random.default_rng(0).normal(size=100)})
    metrics = check_autocorrelation(data, save_path=None, acf_pacf_lags=5, lb_lags=5, lb_auto_lag=False)
    assert metrics.shape == (1, 4)
    assert list(tmp_path.iterdir()) == []
