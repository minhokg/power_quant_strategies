import json

import pandas as pd

from power_quant_strategies.utils.fit_model.fit_xgboost import tune_hyperparameter_xgboost as module


def _split(trading_data):
    data = trading_data.drop(columns="preds")
    return data.iloc[:60].copy(), data.iloc[60:80].copy(), data.iloc[80:].copy()


def test_runs_trials_and_saves_best_params(trading_data, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "get_usable_cpu_count", lambda: 1)

    study, best_params = module.tune_hyperparameter_xgboost(
        *_split(trading_data),
        column_name_target="target",
        n_estimators=5,
        verbose=False,
        n_trials=2,
        show_progress_bar=False,
        save_path=str(tmp_path),
    )

    assert len(study.trials) == 2
    assert list(best_params.columns) == ["Value"]
    assert {"eta", "max_depth", "subsample"} <= set(best_params.index)
    # every trial records its train/valid/test metrics
    assert "pnl_test" in study.trials[0].user_attrs

    assert json.loads((tmp_path / "best_params.json").read_text()) == best_params["Value"].to_dict()
    assert pd.read_parquet(tmp_path / "trials.parquet").shape[0] == 2
    assert (tmp_path / "trials.xlsx").exists()
    assert (tmp_path / "optuna_study.pkl").exists()


def test_slow_trials_are_skipped_unless_all_are_slow(trading_data, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "get_usable_cpu_count", lambda: 1)
    monkeypatch.chdir(tmp_path)
    kwargs = {"column_name_target": "target", "n_estimators": 5, "verbose": False, "n_trials": 2, "show_progress_bar": False, "save_path": None}

    # no trial finishes in 0 seconds: the fastest one is chosen instead
    study, best_params = module.tune_hyperparameter_xgboost(*_split(trading_data), max_run_time_per_model_fit=0.0, **kwargs)

    trials = study.trials_dataframe()
    fastest = study.trials[trials["duration"].idxmin()]
    assert best_params["Value"].to_dict() == fastest.params
    # trials write nothing to the working directory when save_path is None
    assert list(tmp_path.iterdir()) == []
