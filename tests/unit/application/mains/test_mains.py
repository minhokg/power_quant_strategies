import pandas as pd
import pytest

from power_quant_strategies.application.mains import main_feature_importance_and_selection as feature_module
from power_quant_strategies.application.mains import main_train_walk_forward as walk_forward_module
from power_quant_strategies.application.mains import main_tune_hyperparameter as tune_module
from power_quant_strategies.application.settings.models import BaseSettings


@pytest.fixture
def settings(tmp_path) -> BaseSettings:
    return BaseSettings(save_path=str(tmp_path), n_estimators=7, early_stopping_rounds=3, random_state=1, device="cuda")


@pytest.fixture(autouse=True)
def no_logging_setup(monkeypatch):
    for module in (feature_module, tune_module):
        monkeypatch.setattr(module, "setup_logging", lambda: None)


def test_feature_selection_passes_results_on(monkeypatch, settings, tmp_path):
    importance = pd.DataFrame({"score": [1.0]})
    calls = {}
    monkeypatch.setattr(feature_module, "calculate_feature_importance", lambda data, save_path: calls.setdefault("importance", save_path) and importance)
    monkeypatch.setattr(feature_module, "select_features", lambda **kwargs: calls.setdefault("select", kwargs) and ["a"])

    result = feature_module.main_feature_importance_and_selection(pd.DataFrame({"a": [1]}), settings)

    assert result == ["a"]
    assert calls["importance"] == str(tmp_path / "feature_selection")
    assert calls["select"]["feature_importance"] is importance
    assert calls["select"]["feature_selection_strategy"] == settings.feature_selection_strategy


def test_tune_hyperparameter_splits_and_returns_best_params(monkeypatch, settings, trading_data):
    seen = {}

    def fake_tune(**kwargs):
        seen.update(kwargs)
        return None, pd.DataFrame({"Value": {"eta": 0.1, "max_depth": 4}})

    monkeypatch.setattr(tune_module, "tune_hyperparameter_xgboost", fake_tune)
    data = trading_data.reset_index(names="time").set_index("time")
    data.index = data.index.astype(str)  # not a DatetimeIndex yet
    original_index = data.index.copy()

    result = tune_module.main_tune_hyperparameter(data, settings)

    assert result == {"eta": 0.1, "max_depth": 4}
    assert (len(seen["data_train"]), len(seen["data_valid"]), len(seen["data_test"])) == (57, 19, 20)  # 60/20/20 of 96
    assert isinstance(seen["data_train"].index, pd.DatetimeIndex)
    assert seen["device"] == "cuda"
    assert data.index.equals(original_index)  # the caller's data is not changed


def test_train_walk_forward_uses_settings(monkeypatch, settings, trading_data, tmp_path):
    seen = {}

    def fake_walk_forward(**kwargs):
        seen.update(kwargs)
        return ("data_test_all",) + (None,) * 12

    plotted = []
    monkeypatch.setattr(walk_forward_module, "fit_xgboost_walk_forward", fake_walk_forward)
    monkeypatch.setattr(walk_forward_module, "evaluate_trading_plots", lambda data, save_path: plotted.append((data, save_path)))

    walk_forward_module.main_train_walk_forward(trading_data, settings, hyperparameter={"max_depth": 3})

    save_path = tmp_path / "train_walk_forward"
    assert {p.name for p in save_path.iterdir()} == {"01_explore_data", "02_fit_model", "03_evaluate_model"}
    assert (seen["n_estimators"], seen["early_stopping_rounds"], seen["random_state"], seen["device"]) == (7, 3, 1, "cuda")
    assert seen["hyperparameter"] == {"max_depth": 3}
    assert plotted == [("data_test_all", str(save_path))]
