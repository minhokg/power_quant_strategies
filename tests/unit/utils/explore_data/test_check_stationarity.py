import os

import numpy as np
import pandas as pd
import pytest

from power_quant_strategies.utils.explore_data.check_stationarity import check_stationarity


@pytest.fixture
def series() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    noise = rng.normal(size=500)
    return pd.DataFrame({"white_noise": noise, "random_walk": np.cumsum(rng.normal(size=500))})


def test_verdicts_for_stationary_and_non_stationary_series(series, tmp_path):
    metrics = check_stationarity(series, save_path=str(tmp_path))

    assert list(metrics.index) == ["white_noise", "random_walk"]
    assert metrics.loc["white_noise", "adf_verdict"] == "stationary"
    assert metrics.loc["white_noise", "pp_verdict"] == "stationary"
    assert metrics.loc["white_noise", "kpss_verdict"] == "likely stationary"
    assert metrics.loc["random_walk", "adf_verdict"] == "likely non-stationary"
    assert metrics.loc["random_walk", "kpss_verdict"] == "non-stationary"


def test_saves_metrics(series, tmp_path):
    check_stationarity(series, save_path=str(tmp_path))
    folder = tmp_path / "01_explore_data"
    assert (folder / "stationarity_metrics.parquet").exists()
    assert (folder / "stationarity_metrics.xlsx").exists()
    assert pd.read_parquet(folder / "stationarity_metrics.parquet").shape[0] == 2


def test_no_files_without_save_path(series, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    check_stationarity(series[["white_noise"]], save_path=None)
    assert os.listdir(tmp_path) == []


def test_ignores_nans(series):
    series.iloc[::10, 0] = np.nan
    metrics = check_stationarity(series[["white_noise"]], save_path=None)
    assert metrics.loc["white_noise", "adf_number_of_observations"] < 500
