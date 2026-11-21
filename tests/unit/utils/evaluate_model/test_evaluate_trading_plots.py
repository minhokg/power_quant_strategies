import os

import pandas as pd
import pytest

from power_quant_strategies.utils.evaluate_model.evaluate_trading_plots import evaluate_trading_plots


def test_saves_pnl_data_and_plots(trading_data, tmp_path):
    data = trading_data.assign(
        threshold_train_valid=0.5,
        spread_realized=(2 * trading_data["target"] - 1) * trading_data["sample_weight"],
    )

    evaluate_trading_plots(data, save_path=str(tmp_path))

    folder = tmp_path / "03_evaluate_model"
    assert os.path.exists(folder / "walk_forward_cumulative_pnl.png")
    assert os.path.exists(folder / "walk_forward_cumulative_pnl_last_year.png")

    result = pd.read_parquet(folder / "data_test_all_with_pnl.parquet")
    assert set(result["trading_direction"]) == {1, -1}
    assert result["pnl_cumulative"].iloc[-1] == pytest.approx(result["pnl"].sum())
    # always long and always short mirror each other
    assert (result["pnl_always_long_cumulative"] == -result["pnl_always_short_cumulative"]).all()
