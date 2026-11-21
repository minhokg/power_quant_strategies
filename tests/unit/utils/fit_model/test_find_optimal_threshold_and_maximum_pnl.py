import pandas as pd
import pytest

from power_quant_strategies.utils.fit_model.find_optimal_threshold_and_maximum_pnl import find_optimal_threshold_and_maximum_pnl


def test_threshold_separates_winning_longs_from_winning_shorts(results_dir):
    # high scores have positive spreads, low scores negative: the best cut is between 0.4 and 0.6
    data = pd.DataFrame(
        {"preds": [0.1, 0.2, 0.4, 0.6, 0.8, 0.9], "target": [0, 0, 0, 1, 1, 1], "sample_weight": [1.0] * 6},
        index=pd.date_range("2026-01-01", periods=6, freq="h"),
    )

    threshold, maximum_pnl = find_optimal_threshold_and_maximum_pnl(data, column_name_target="target", split="train", save_path=results_dir)

    assert threshold == 0.4
    # long 0.6, 0.8, 0.9 and short 0.1, 0.2 each win 1 (one day). The row at the threshold counts as both long
    # and short, so it adds nothing.
    assert maximum_pnl == pytest.approx(5.0)

    saved = pd.read_parquet(f"{results_dir}/02_fit_model/threshold_and_maximum_pnl_train.parquet")
    assert saved.loc["threshold", "Value"] == threshold
    assert (pd.Index(["threshold", "maximum_pnl_per_day"]) == saved.index).all()


def test_pnl_is_per_day():
    one_day = pd.DataFrame(
        {"preds": [0.1, 0.2, 0.8, 0.9], "target": [0, 0, 1, 1], "sample_weight": [1.0, 2.0, 3.0, 4.0]},
        index=pd.date_range("2026-01-01 08:00", periods=4, freq="h"),
    )
    two_days = one_day.set_axis(pd.to_datetime(["2026-01-01 08:00", "2026-01-01 09:00", "2026-01-02 08:00", "2026-01-02 09:00"]))

    threshold_1, pnl_1 = find_optimal_threshold_and_maximum_pnl(one_day, column_name_target="target", split="train", save_path=None)
    threshold_2, pnl_2 = find_optimal_threshold_and_maximum_pnl(two_days, column_name_target="target", split="train", save_path=None)

    assert threshold_1 == threshold_2
    assert pnl_2 == pytest.approx(pnl_1 / 2)


def test_does_not_change_input(trading_data):
    before = trading_data.copy()
    find_optimal_threshold_and_maximum_pnl(trading_data, column_name_target="target", split="test", save_path=None)
    pd.testing.assert_frame_equal(trading_data, before)
