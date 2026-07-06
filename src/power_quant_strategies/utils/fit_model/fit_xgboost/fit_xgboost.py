import logging
from typing import Union

import numpy as np
import pandas as pd

from xgboost import XGBClassifier


def fit_xgboost(
    data_train: pd.DataFrame,
    data_valid: pd.DataFrame,
    data_test: pd.DataFrame,
    column_name_target: str,
    n_estimators: int = 500,
    early_stopping_rounds: int = 10,
    verbose: Union[bool, int, None] = True,
    hyperparameter: dict = None,
    random_state: int = 42,
    device: str = "cpu",
) -> tuple:
    """
    Fit XGBoost model.

    :param data_train: Train dataset
    :param data_valid: Validation dataset
    :param data_test: Test dataset
    :param column_name_target: name of the target column
    :param n_estimators: number of estimators
    :param early_stopping_rounds: early stopping rounds
    :param verbose: verbosity level
    :param hyperparameter: hyperparameter
    :param random_state: random seed
    :param device: Device to use for training
    :return: tuple with model and original datasets
    """
    # give user feedback
    logging.info("fit xgboost")

    # For some reason this sometimes gets converted to a float (still .000) during backtesting and then throws an error inside XGBoost
    hyperparameter["max_depth"] = int(hyperparameter["max_depth"])

    # determine feature names
    features = data_train.drop(columns=[col for col in [column_name_target, "preds"] if col in data_train.columns]).columns.tolist()

    # create model object instance
    model = XGBClassifier(**hyperparameter, n_estimators=n_estimators, early_stopping_rounds=early_stopping_rounds, random_state=random_state, device=device)

    # extract target
    target_train = data_train[[column_name_target]]
    target_valid = data_valid[[column_name_target]]

    # extract sample weights
    sample_weight_train = data_train[["sample_weight"]]
    sample_weight_valid = data_valid[["sample_weight"]]

    # make sure sample weights are not zero as otherwise fit_xgboost will complain
    sample_weight_train = np.maximum(sample_weight_train, 1e-6)
    sample_weight_valid = np.maximum(sample_weight_valid, 1e-6)

    # extract features
    features_train = data_train[features]
    features_valid = data_valid[features]
    features_test = data_test[features]

    # fit model
    model.fit(
        X=features_train,
        y=target_train,
        sample_weight=sample_weight_train,
        eval_set=[(features_valid, target_valid)],
        sample_weight_eval_set=[sample_weight_valid],
        verbose=verbose,
    )

    # predict
    preds_train = model.predict_proba(X=features_train)[:, 1]
    preds_valid = model.predict_proba(X=features_valid)[:, 1]
    preds_test = model.predict_proba(X=features_test)[:, 1]

    # add to datasets
    data_train.loc[:, "preds"] = preds_train
    data_valid.loc[:, "preds"] = preds_valid
    data_test.loc[:, "preds"] = preds_test

    return model, data_train, data_valid, data_test
