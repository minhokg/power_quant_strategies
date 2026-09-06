import json
import logging
import os

import pandas as pd


def _cap_mask_by_score(
    classified_features: pd.DataFrame,
    mask_to_keep: pd.Series,
    max_features: int,
    feature_selection_strategy: str,
) -> pd.Series:
    """
    Cap selected features by score and log if truncation happens.

    :param classified_features: Classified feature importance DataFrame.
    :param mask_to_keep: Boolean mask of selected features.
    :param max_features: Maximum number of features to keep.
    :param feature_selection_strategy: Strategy used for selection.
    :return: Boolean mask capped to the highest-scoring selected features
    """
    n_selected = int(mask_to_keep.sum())

    if max_features is None or n_selected <= max_features:
        return mask_to_keep

    logging.warning(
        "Truncating features: selected=%s > max_features=%s using strategy=%s",
        n_selected,
        max_features,
        feature_selection_strategy,
    )

    selected_features_ranked = classified_features.loc[mask_to_keep].sort_values("score", ascending=False)

    selected_indices = selected_features_ranked.index[:max_features]

    return classified_features.index.to_series().isin(selected_indices)


def select_features(feature_importance: pd.DataFrame, feature_selection_strategy: str = "boruta_style_aggressive", max_features: int = 350, save_path: str = os.getcwd()) -> list:
    """
    Select features.

    :param feature_importance: Result from calculate_feature_importance
    :param feature_selection_strategy: Feature selection mode. Use "kneedle" to select features up to the kneww point of
    the importance score curve (adaptive cutoff based on diminishing returns), "boruta_style_aggressive" to keep only
    Confirmed features, or "boruta_style_defensive" to keep Confirmed and Tentative features.
    :param max_features: Maximum number of features to keep
    :param save_path: Directory to save the features to keep and features to drop
    :return: Tuple of features_to_keep
    """
    # give user feedback
    logging.info("select features")

    required_columns = ["boruta_like_result", "score"]
    missing_columns = [column_name for column_name in required_columns if column_name not in feature_importance.columns]
    if missing_columns:
        raise ValueError(f"feature_importance is missing required columns: {missing_columns}")

    # consider only features that have a classification
    classified_features = feature_importance[feature_importance["boruta_like_result"].notna()]

    if classified_features.empty:
        raise ValueError("No classified features found in feature_importance")

    # keep features depending on the selection strategy
    # - boruta_style_aggressive: keep boruta_like_result = Confirmed
    # - boruta_style_defensive: keep boruta_like_result = Confirmed or Tentative
    if feature_selection_strategy == "boruta_style_aggressive":
        mask_to_keep = classified_features["boruta_like_result"].isin({"Confirmed"})
    elif feature_selection_strategy == "boruta_style_defensive":
        mask_to_keep = classified_features["boruta_like_result"].isin({"Confirmed", "Tentative"})

    mask_to_keep = _cap_mask_by_score(
        classified_features=classified_features,
        mask_to_keep=mask_to_keep,
        max_features=max_features,
        feature_selection_strategy=feature_selection_strategy,
    )

    features_to_keep = sorted(classified_features.index[mask_to_keep].tolist())

    logging.info(
        "selected %s features using %s strategy",
        len(features_to_keep),
        feature_selection_strategy,
    )

    # save to disk
    os.makedirs(os.path.join(save_path, "01_explore_data"), exist_ok=True)
    file_path = os.path.join(save_path, "01_explore_data", "feature_selection_features_to_keep.json")
    logging.info("saving selected features to %s", file_path)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write("[\n" + ",\n".join(json.dumps(item, ensure_ascii=False) for item in features_to_keep) + "\n]")

    return features_to_keep
