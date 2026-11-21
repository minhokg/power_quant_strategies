import numpy as np
import pandas as pd
import pytest

from power_quant_strategies.utils.evaluate_model.evaluate_trading_metrics import evaluate_trading_metrics


def _data(preds, target, sample_weight) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=len(preds), freq="h")
    return pd.DataFrame({"preds": preds, "target": target, "sample_weight": sample_weight}, index=index)


def test_hand_computed_pnl():
    # threshold 0.5: preds 0.9 -> long with bet size 1, preds 0.1 -> short with bet size 1
    data = _data(preds=[0.9, 0.1], target=[1, 0], sample_weight=[10.0, 4.0])

    metrics = evaluate_trading_metrics(data, column_name_target="target", split="test", threshold=0.5, save_path=None)["Value"]

    assert metrics["n_time_slices"] == 2
    assert metrics["pnl_long"] == pytest.approx(10.0)  # spread +10, long
    assert metrics["pnl_short"] == pytest.approx(4.0)  # spread -4, short
    assert metrics["pnl"] == pytest.approx(14.0)
    assert metrics["win_rate"] == 1.0
    assert metrics["max_draw_down"] == 0.0
    assert metrics["profit_factor"] == np.inf  # no losing trade


def test_losing_trades_lower_profit_factor_and_cause_drawdown():
    data = _data(preds=[0.9, 0.9, 0.1], target=[1, 0, 0], sample_weight=[6.0, 3.0, 1.0])

    metrics = evaluate_trading_metrics(data, column_name_target="target", split="test", threshold=0.5, save_path=None)["Value"]

    assert metrics["pnl"] == pytest.approx(6.0 - 3.0 + 1.0)
    assert metrics["profit_factor"] == pytest.approx(7.0 / 3.0)
    assert metrics["win_rate"] == pytest.approx(2 / 3)
    assert metrics["max_draw_down"] == pytest.approx(3.0)


def test_metrics_are_sorted_and_saved(trading_data, results_dir):
    metrics = evaluate_trading_metrics(trading_data, column_name_target="target", split="valid", threshold=0.5, save_path=results_dir)

    assert list(metrics.index) == sorted(metrics.index)
    assert list(metrics.columns) == ["Value"]
    assert metrics.loc["pnl", "Value"] > 0  # the scores in the fixture are informative
    saved = pd.read_parquet(f"{results_dir}/03_evaluate_model/trading_metrics_valid.parquet")
    pd.testing.assert_frame_equal(saved, metrics)


def test_does_not_change_input(trading_data):
    before = trading_data.copy()
    evaluate_trading_metrics(trading_data, column_name_target="target", split="test", threshold=0.5, save_path=None)
    pd.testing.assert_frame_equal(trading_data, before)
