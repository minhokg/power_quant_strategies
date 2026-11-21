from power_quant_strategies.utils.fit_model.fit_xgboost.fit_xgboost import fit_xgboost

HYPERPARAMETER = {"max_depth": 3.0, "eta": 0.3}


def _split(trading_data):
    data = trading_data.drop(columns="preds")
    return data.iloc[:60].copy(), data.iloc[60:80].copy(), data.iloc[80:].copy()


def test_adds_probabilities_to_every_split(trading_data):
    data_train, data_valid, data_test = _split(trading_data)

    model, data_train, data_valid, data_test = fit_xgboost(data_train, data_valid, data_test, column_name_target="target", n_estimators=10, verbose=False, hyperparameter=HYPERPARAMETER)

    for data in (data_train, data_valid, data_test):
        assert data["preds"].between(0, 1).all()
    # target and preds are not used as features
    assert list(model.get_booster().feature_names) == ["sample_weight", "feature_a", "feature_b"]
    assert model.get_params()["max_depth"] == 3


def test_does_not_change_the_hyperparameter_dict(trading_data):
    hyperparameter = dict(HYPERPARAMETER)
    fit_xgboost(*_split(trading_data), column_name_target="target", n_estimators=5, verbose=False, hyperparameter=hyperparameter)
    assert hyperparameter == HYPERPARAMETER


def test_works_without_hyperparameters(trading_data):
    _, data_train, _, _ = fit_xgboost(*_split(trading_data), column_name_target="target", n_estimators=5, verbose=False)
    assert "preds" in data_train.columns
