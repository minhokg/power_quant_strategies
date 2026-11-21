import os

from power_quant_strategies.utils.evaluate_model.check_model_score_vs_spread_dependency import check_model_score_vs_spread_dependency


def test_saves_plot(trading_data, results_dir):
    check_model_score_vs_spread_dependency(trading_data, column_name_target="target", split="test", save_path=results_dir)
    assert os.path.exists(os.path.join(results_dir, "03_evaluate_model", "model_score_vs_spread_test.png"))


def test_no_plot_without_save_path(trading_data, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = trading_data.copy()
    check_model_score_vs_spread_dependency(trading_data, column_name_target="target", split="test", save_path=None)
    assert list(tmp_path.iterdir()) == []
    assert trading_data.equals(before)
