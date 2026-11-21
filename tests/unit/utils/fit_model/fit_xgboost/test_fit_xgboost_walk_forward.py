import numpy as np
import pandas as pd
import pytest

from power_quant_strategies.utils.fit_model.fit_xgboost import fit_xgboost_walk_forward as module


@pytest.fixture
def daily_data() -> pd.DataFrame:
    """Return 8 days of hourly data whose target follows feature_a."""
    rng = np.random.default_rng(0)
    index = pd.date_range("2026-01-01", periods=8 * 24, freq="h")
    feature = rng.normal(size=len(index))
    return pd.DataFrame(
        {"feature_a": feature, "target": (feature > 0).astype(int), "sample_weight": rng.uniform(1, 5, len(index))},
        index=index,
    )


def test_every_day_after_the_window_is_tested_once(daily_data, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "get_usable_cpu_count", lambda: 1)
    for folder in ("02_fit_model", "03_evaluate_model"):
        (tmp_path / folder).mkdir()

    result = module.fit_xgboost_walk_forward(
        data=daily_data.sample(frac=1, random_state=0),  # must be sorted by the function itself
        column_name_target="target",
        hyperparameter={"max_depth": 2},
        n_days_sliding_window=5,
        n_estimators=5,
        save_path=str(tmp_path),
    )
    data_test_all, threshold_all, maximum_pnl_all = result[:3]

    # days 6, 7 and 8 are predicted, each by a model trained on the 5 days before it
    assert sorted(data_test_all.index.normalize().unique()) == list(pd.date_range("2026-01-06", periods=3, freq="D"))
    assert len(data_test_all) == 3 * 24
    assert {"preds", "threshold_train_valid", "spread_realized", "spread_expected"} <= set(data_test_all.columns)
    assert threshold_all.shape == (3, 3)
    assert maximum_pnl_all.shape == (3, 3)
    assert (tmp_path / "03_evaluate_model" / "trading_metrics_test.parquet").exists()


def test_needs_more_days_than_the_window(daily_data):
    with pytest.raises(AssertionError):
        module.fit_xgboost_walk_forward(data=daily_data, column_name_target="target", hyperparameter={}, n_days_sliding_window=8, save_path=None)
