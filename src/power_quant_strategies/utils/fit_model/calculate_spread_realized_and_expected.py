import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from power_quant_strategies.utils.helper.remove_nans import remove_nans_iteratively_dataframe


def _fit_spread_expected(
    data: pd.DataFrame,
    target: pd.Series,
) -> tuple[RidgeCV | None, pd.Index]:
    """
    Fit expected spread (for now hardcoded as linear function, maybe  .

    we use a linear fit because we trust it is good enough, and it does not need hyperparameter tuning

    :param data: input data date x features
    :param target: realized spread
    :return: fitted model and column mask of available features
    """
    reduced_data = remove_nans_iteratively_dataframe(data)

    model = None
    if reduced_data.empty:
        logging.warning("No valid dates found in data set to fit linear regression for spread_expected.")
    else:
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=(0.1, 1.0, 10.0)))

        model.fit(X=reduced_data, y=target.loc[reduced_data.index])

    # true false mask of available columns (features) in reduced_data
    feature_mask = reduced_data.columns
    return model, feature_mask


def calculate_spread_realized(
    data: pd.DataFrame,
    column_name_target: str,
    required_cols: list[str] | None = None,
) -> pd.Series:
    """
    Calculate realized spread.

    :param data: Original data with columns 'sample_weight' and column_name_target.
    :param column_name_target: Defines the target column name.
    :param required_cols: List of required columns, if None defaults to [column_name_target, 'sample_weight']
    :return: data with new column spread
    """
    # check inputs
    required_cols = [column_name_target, "sample_weight"] if required_cols is None else required_cols
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise KeyError("columns are missing: %s", missing)
    return (2 * data[column_name_target] - 1) * data["sample_weight"]


def calculate_spread_expected(
    model: RidgeCV,
    features: pd.DataFrame,
    feature_mask: pd.Index,
) -> pd.Series:
    """
    Calculate expected spread, returns NaN for data rows containing NaNs.

    :param model: fitted model (see _fit_spread_expected).
    :param features: dates x features.
    :param feature_mask: mask of available features (see _fit_spread_expected).
    :return: expected spread at each date (NaN of appropriate shape if model is None)
    """
    result = pd.Series(data=np.nan, index=features.index, name="spread_expected")

    date_mask = features[feature_mask].notnull().all(axis=1)
    features_masked = features.loc[date_mask, feature_mask]
    if model is not None and not features_masked.empty:
        # mask of dates (rows) without NaNs
        result[date_mask] = model.predict(X=features.loc[date_mask, feature_mask])
    else:
        logging.warning("No valid dates found in data set to fit linear regression for spread_expected.")
    return result


def add_spread_realized_and_expected(
    data_train: pd.DataFrame,
    data_valid: pd.DataFrame,
    data_test: pd.DataFrame,
    column_name_target: str,
) -> tuple:
    """
    Calculate spread realized and expected.

    This will use all columns that are not target, sample_weight or spread_realized as features and regress it on
    spread_realized.
    :param data_train: data train with column_name_target and sample weight
    :param data_valid: data valid with column_name_target and sample weight
    :param data_test: data test with column_name_target and sample weight
    :param column_name_target: target column name
    :return: data_train, data_valid, data_test with new columns spread_realized and spread_expected
    """
    # check if the desired columns are already there
    if any("spread_realized" in df.columns for df in (data_train, data_valid, data_test)):
        logging.warning(msg="spread_realized column is already present. Will skip.")
        return data_train, data_valid, data_test
    if any("spread_expected" in df.columns for df in (data_train, data_valid, data_test)):
        logging.warning(msg="spread_expected column is already present. Will skip.")
        return data_train, data_valid, data_test

    # combine train and valid for fitting spread expected
    data_train["spread_realized"] = calculate_spread_realized(data_train, column_name_target)
    data_valid["spread_realized"] = calculate_spread_realized(data_valid, column_name_target)
    data_test["spread_realized"] = calculate_spread_realized(data_test, column_name_target)
    data_train_valid = pd.concat(objs=[data_train, data_valid], axis=0)

    # define features and target
    features_train = data_train.drop(labels=[column_name_target, "sample_weight", "spread_realized"], axis=1)
    features_valid = data_valid.drop(labels=[column_name_target, "sample_weight", "spread_realized"], axis=1)
    features_test = data_test.drop(labels=[column_name_target, "sample_weight", "spread_realized"], axis=1)
    features_train_valid = data_train_valid.drop(labels=[column_name_target, "sample_weight", "spread_realized"], axis=1)

    # spread expected is obtained from a fit over the train and validation data
    model, feature_mask = _fit_spread_expected(data=features_train_valid, target=data_train_valid["spread_realized"])

    data_train["spread_expected"] = calculate_spread_expected(model=model, features=features_train, feature_mask=feature_mask)
    data_valid["spread_expected"] = calculate_spread_expected(model=model, features=features_valid, feature_mask=feature_mask)
    data_test["spread_expected"] = calculate_spread_expected(model=model, features=features_test, feature_mask=feature_mask)

    return data_train, data_valid, data_test
