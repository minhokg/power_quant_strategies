import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from power_quant_strategies.utils.helper.get_usable_cpu_count import get_usable_cpu_count


def _parse_xgb_tree_features(trees_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many trees use each feature for splitting.

    :param trees_dataframe: Result of booster.trees_to_dataframe().
    :return: Count of trees in which each feature is used for splitting.
    """
    split_rows = trees_dataframe[trees_dataframe["Feature"] != "Leaf"]
    if split_rows.empty:
        return pd.Series(dtype=float)

    return split_rows.groupby("Feature")["Tree"].nunique()


def _train_single_xgb_model_and_collect_feature_stats(
    features_train: pd.DataFrame,
    target_train: pd.Series,
    weights_train: pd.Series | None,
    seed: int,
    xgb_params: dict[str, Any],
) -> dict[str, Any]:
    """
    Train a single XGBoost model and collect feature usage statistics.

    :param features_train: Training features.
    :param target_train: Training target.
    :param weights_train: Training sample weights.
    :param seed: Random seed.
    :param xgb_params: XGBoost model parameters.
    :return: Dictionary containing total gain, split count, tree usage count,
    and number of trees.
    """
    model = xgb.XGBClassifier(
        random_state=seed,
        **xgb_params,
    )

    model.fit(
        X=features_train,
        y=target_train,
        sample_weight=weights_train,
    )

    booster = model.get_booster()

    if booster.feature_names is None:
        raise ValueError("xgboost booster returned no feature names")

    if list(features_train.columns) != booster.feature_names:
        raise ValueError("feature order mismatch")

    trees_dataframe = booster.trees_to_dataframe()

    return {
        "total_gain": booster.get_score(importance_type="total_gain"),
        "weight": booster.get_score(importance_type="weight"),
        "tree_feature_counts": _parse_xgb_tree_features(trees_dataframe),
        "model_num_trees": trees_dataframe["Tree"].nunique(),
    }


def _create_fake_features(
    real_features: pd.DataFrame,
    usable_real_feature_names: list[str],
    fake_feature_prefix: str,
    random_state: int,
) -> pd.DataFrame:
    """
    Create fake features by permuting real feature values.

    This preserves marginal distributions, missing-value patterns, and scales,
    while destroying any relationship to the target.

    :param real_features: Real feature matrix.
    :param usable_real_feature_names: Usable real feature names.
    :param fake_feature_prefix: Prefix for generated fake features.
    :param random_state: Random seed.
    :return: Dataframe containing one fake feature per usable real feature.
    """
    random_generator = np.random.default_rng(random_state)

    fake_feature_dict: dict[str, pd.Series] = {}

    for feature_name in usable_real_feature_names:
        fake_feature_name = f"{fake_feature_prefix}{feature_name}"

        shuffled_values = random_generator.permutation(real_features[feature_name].to_numpy(copy=True))

        fake_feature_dict[fake_feature_name] = pd.Series(
            shuffled_values,
            index=real_features.index,
        )

    return pd.DataFrame(fake_feature_dict)


def calculate_feature_importance(
    data: pd.DataFrame,
    column_name_target: str = "target",
    column_name_sample_weight: str = "sample_weight",
    n_splits: int = 5,
    n_seeds: int = 5,
    save_path: str | None = os.getcwd(),
    fake_feature_prefix: str = "fake_",
    fake_feature_random_state: int = 42,
) -> pd.DataFrame:
    """
    Calculate feature importance with XGBoost and fake-feature calibration.

    The method trains XGBoost models across time-series splits and random seeds,
    computes a stabilised feature score, and classifies features into
    Confirmed, Tentative, and Rejected based on the score distribution of fake features.

    Features that are unusable for modelling are excluded from training, then added back into
    the final output with no rank and a Rejected label.

    :param data: Dataframe containing target, sample weight, and feature columns.
    :param column_name_target: Name of the target column.
    :param column_name_sample_weight: Name of the sample weight column.
    :param n_splits: Number of time-series cross-validation splits.
    :param n_seeds: Number of random seeds.
    :param save_path: Directory in which to save the result files, or None to
        skip saving.
    :param fake_feature_prefix: Prefix for generated fake features.
    :param fake_feature_random_state: Random seed for fake feature generation.
    :return: Dataframe with final feature importance ranking and boruta-like
        classification for all original real features.
    """
    logging.info("calculate feature importance")

    data_copy = data.copy()

    # --- INPUT VALIDATION ---
    if column_name_target not in data_copy.columns:
        raise ValueError("target column missing")

    if column_name_sample_weight not in data_copy.columns:
        raise ValueError("sample_weight column missing")

    if n_splits < 1 or n_seeds < 1:
        raise ValueError("n_splits and n_seeds must be >= 1")

    if len(data_copy) <= n_splits:
        raise ValueError("data must contain more rows than n_splits")

    target = data_copy[column_name_target]
    sample_weight = data_copy[column_name_sample_weight]

    if target.isna().any():
        raise ValueError("target contains NaNs")

    if not set(target.unique()).issubset({0, 1}):
        raise ValueError("target must be binary")

    if sample_weight.isna().any() or (sample_weight < 0).any():
        raise ValueError("invalid sample_weight")

    target = target.astype(int)
    sample_weight = sample_weight.astype(float)

    real_features = data_copy.drop(columns=[column_name_target, column_name_sample_weight]).copy()

    if real_features.empty:
        raise ValueError("no features")

    if real_features.columns.duplicated().any():
        raise ValueError("duplicate feature names found")

    if any(name.startswith(fake_feature_prefix) for name in real_features.columns):
        raise ValueError("feature names collide with fake feature prefix")

    # --- DATA CLEANING ---
    # XGBoost handles NaN but not +/-inf, so replace infinities with NaN.
    numeric_columns = real_features.select_dtypes(include=[np.number]).columns
    real_features.loc[:, numeric_columns] = real_features.loc[:, numeric_columns].replace([np.inf, -np.inf], np.nan)

    # A feature is usable for modelling if it contains at least one non-missing
    # value after cleaning. Features failing this check are excluded from model
    # fitting, then added back into the final output as unranked Rejected
    # features.
    usable_feature_mask = real_features.notna().mean() > 0
    usable_real_feature_names = real_features.columns[usable_feature_mask].tolist()
    unusable_real_feature_names = real_features.columns[~usable_feature_mask].tolist()

    if not usable_real_feature_names:
        raise ValueError("no usable features")

    logging.info(
        "usable real features for xgboost: %s / %s",
        len(usable_real_feature_names),
        len(real_features.columns),
    )

    if unusable_real_feature_names:
        logging.info(
            "excluded unusable real features from modelling: %s",
            len(unusable_real_feature_names),
        )

    usable_real_features = real_features.loc[:, usable_real_feature_names].copy()

    # --- FAKE FEATURES ---
    # Create one fake feature per usable real feature.
    # This keeps the noise baseline comparable to the real feature space.
    fake_features = _create_fake_features(
        real_features=usable_real_features,
        usable_real_feature_names=usable_real_feature_names,
        fake_feature_prefix=fake_feature_prefix,
        random_state=fake_feature_random_state,
    )

    model_features = pd.concat([usable_real_features, fake_features], axis=1)
    model_feature_names = model_features.columns.tolist()

    logging.info(
        "total model features after adding fake features: %s",
        len(model_feature_names),
    )

    # --- storage initialization ---
    model_usage_count = dict.fromkeys(model_feature_names, 0)
    tree_usage_count = dict.fromkeys(model_feature_names, 0)
    total_gain_sum = dict.fromkeys(model_feature_names, 0)
    total_split_count = dict.fromkeys(model_feature_names, 0)

    # --- model parameters ---
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 300,
        "learning_rate": 0.076,
        "max_depth": 5,
        "min_child_weight": 6,
        "gamma": 0.75,
        "subsample": 0.83,
        "colsample_bytree": 0.73,
        "colsample_bylevel": 0.76,
        "reg_alpha": 0.14,
        "reg_lambda": 1.0,
        "max_delta_step": 5,
        "tree_method": "hist",
        "n_jobs": get_usable_cpu_count(),
    }

    time_series_split = TimeSeriesSplit(n_splits=n_splits)
    seeds = list(range(n_seeds))

    total_expected_models = n_splits * n_seeds
    logging.info(
        "running feature importance models: %s splits x %s seeds = %s models",
        n_splits,
        n_seeds,
        total_expected_models,
    )

    total_models = 0
    total_trees = 0
    skipped_models = 0

    # training loop
    for split_index, (train_indices, _) in enumerate(time_series_split.split(model_features), start=1):
        features_train = model_features.iloc[train_indices]
        target_train = target.iloc[train_indices]
        weights_train = sample_weight.iloc[train_indices]

        # early folds can contain only one class, which XGBoost classification
        # cannot fit on. Skip those folds.
        if target_train.nunique() < 2:
            skipped_models += len(seeds)
            logging.warning(
                "skipping split %s/%s due tp single class",
                split_index,
                n_splits,
            )
            continue

        for seed_index, seed in enumerate(seeds, start=1):
            total_models += 1

            logging.info(
                "training model %s/%s (split %s/%s, seed %s/%s)",
                total_models,
                total_expected_models - skipped_models,
                split_index,
                n_splits,
                seed_index,
                n_seeds,
            )

            model_stats = _train_single_xgb_model_and_collect_feature_stats(
                features_train=features_train,
                target_train=target_train,
                weights_train=weights_train,
                seed=seed,
                xgb_params=xgb_params,
            )

            total_trees += model_stats["model_num_trees"]

            for feature_name, gain_value in model_stats["total_gain"].items():
                total_gain_sum[feature_name] += float(gain_value)

            for feature_name, split_value in model_stats["weight"].items():
                # total number of splits using a feature across all trees in all models
                total_split_count[feature_name] += int(split_value)

                # number of models in which the feature was used at least once
                model_usage_count[feature_name] += 1

            for feature_name, tree_count in model_stats["tree_feature_counts"].items():
                tree_usage_count[feature_name] += int(tree_count)

    if total_models == 0:
        raise ValueError("no models trained")

    # build feature importance table for modelled features
    feature_importance_modelled = pd.DataFrame(index=model_feature_names)

    feature_importance_modelled["model_usage_rate"] = [model_usage_count[feature_name] / total_models for feature_name in model_feature_names]

    feature_importance_modelled["tree_usage_rate"] = [(tree_usage_count[feature_name] / total_trees if total_trees > 0 else 0.0) for feature_name in model_feature_names]

    feature_importance_modelled["total_split_count"] = [total_split_count[feature_name] for feature_name in model_feature_names]

    feature_importance_modelled["mean_gain_per_split"] = 0.0

    nonzero_split_mask = feature_importance_modelled["total_split_count"] > 0

    feature_importance_modelled.loc[nonzero_split_mask, "mean_gain_per_split"] = [
        total_gain_sum[feature_name] / total_split_count[feature_name] for feature_name in feature_importance_modelled.index[nonzero_split_mask]
    ]

    # core scoring logic
    # raw xgboost gain is noisy and biased:
    # early splits can dominate gain
    # correlated features can swap across runs
    # many weak splits can accumulate large total gain

    # we therefore rank features by a stabilised score:
    # score = model_usage_rate * mean_gain_per_split * log1p(total_split_count)
    # where:
    # model_usage_rate measures consistency across models
    # mean_gain_per_split measures average split strength
    # log1p(total_split_count) rewards repeated use, but only mildly

    # this gives a more stable proxy for feature relevance than raw gain alone.
    feature_importance_modelled["score"] = (
        feature_importance_modelled["model_usage_rate"] * feature_importance_modelled["mean_gain_per_split"] * np.log1p(feature_importance_modelled["total_split_count"])
    )

    # boruta-like classification
    # fake features define a noise distribution.
    # we classify real features by comparing their score against fake-feature
    # scores:
    # confirmed: score > max(fake_scores)
    # tentative: q50(fake_scores) < score <= max(fake_scores)
    # rejected: score <= q50(fake_scores)
    is_fake_feature = feature_importance_modelled.index.str.startswith(fake_feature_prefix)
    fake_scores = feature_importance_modelled.loc[is_fake_feature, "score"].dropna()

    feature_importance_modelled["boruta_like_result"] = pd.NA

    if len(fake_scores) > 0:
        max_fake_score = float(fake_scores.max())
        median_fake_score = float(fake_scores.quantile(0.5))

        is_real_feature = ~is_fake_feature

        feature_importance_modelled.loc[
            is_real_feature & (feature_importance_modelled["score"] > max_fake_score),
            "boruta_like_result",
        ] = "Confirmed"

        feature_importance_modelled.loc[
            is_real_feature & (feature_importance_modelled["score"] > median_fake_score) & (feature_importance_modelled["score"] <= max_fake_score),
            "boruta_like_result",
        ] = "Tentative"

        feature_importance_modelled.loc[
            is_real_feature & (feature_importance_modelled["score"] <= median_fake_score),
            "boruta_like_result",
        ] = "Rejected"

    # Fake features are only needed for calibration and are not part of the
    # final output.
    feature_importance_modelled = feature_importance_modelled.loc[~is_fake_feature].copy()

    # --- FINAL RANKING FOR MODELLED REAL FEATURES ---
    # Recompute score shares on real features only.
    total_score_modelled_real_features = feature_importance_modelled["score"].sum()

    feature_importance_modelled["score_share"] = feature_importance_modelled["score"] / total_score_modelled_real_features if total_score_modelled_real_features > 0 else np.nan

    feature_importance_modelled = feature_importance_modelled.sort_values(
        by=[
            "score",
            "model_usage_rate",
            "mean_gain_per_split",
            "total_split_count",
        ],
        ascending=False,
    )

    feature_importance_modelled["rank"] = range(1, len(feature_importance_modelled) + 1)
    feature_importance_modelled["cumulative_score_share"] = feature_importance_modelled["score_share"].cumsum()

    # --- ADD BACK UNUSABLE REAL FEATURES ---
    # Features excluded from modelling are returned as unranked Rejected
    # features so that the final output still covers all original real features.
    if unusable_real_feature_names:
        feature_importance_unusable = pd.DataFrame(index=unusable_real_feature_names)
        feature_importance_unusable["rank"] = pd.Series(
            pd.array([pd.NA] * len(feature_importance_unusable), dtype="Int64"),
            index=feature_importance_unusable.index,
        )
        feature_importance_unusable["score"] = 0.0
        feature_importance_unusable["score_share"] = 0.0
        feature_importance_unusable["cumulative_score_share"] = np.nan
        feature_importance_unusable["boruta_like_result"] = "Rejected"
        feature_importance_unusable["model_usage_rate"] = 0.0
        feature_importance_unusable["tree_usage_rate"] = 0.0
        feature_importance_unusable["total_split_count"] = 0
        feature_importance_unusable["mean_gain_per_split"] = 0.0

        feature_importance = pd.concat(
            [feature_importance_modelled, feature_importance_unusable],
            axis=0,
        )
    else:
        feature_importance = feature_importance_modelled.copy()

    # --- FINAL COLUMN ORDER ---
    feature_importance = feature_importance[
        [
            "rank",
            "score",
            "score_share",
            "cumulative_score_share",
            "boruta_like_result",
            "model_usage_rate",
            "tree_usage_rate",
            "total_split_count",
            "mean_gain_per_split",
        ]
    ]

    # Keep ranked/modelled features first, then unranked/unusable features.
    feature_importance = feature_importance.sort_values(
        by=["rank", "score"],
        ascending=[True, False],
        na_position="last",
    )

    # --- SAVE ---
    if save_path is not None:
        output_path = os.path.join(save_path, "01_explore_data")
        os.makedirs(output_path, exist_ok=True)

        feature_importance.to_parquet(
            os.path.join(output_path, "feature_importance.parquet"),
            index=True,
        )

        feature_importance.to_excel(
            os.path.join(output_path, "feature_importance.xlsx"),
        )

    return feature_importance
