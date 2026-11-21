import json

import pandas as pd
import pytest

from power_quant_strategies.utils.preprocess_data.select_features import select_features


@pytest.fixture
def feature_importance() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "boruta_like_result": ["Confirmed", "Confirmed", "Tentative", "Rejected", None],
            "score": [5.0, 3.0, 2.0, 1.0, 0.0],
        },
        index=["b_confirmed", "a_confirmed", "tentative", "rejected", "unclassified"],
    )


def test_aggressive_keeps_confirmed(feature_importance, tmp_path):
    selected = select_features(feature_importance, "boruta_style_aggressive", save_path=str(tmp_path))
    assert selected == ["a_confirmed", "b_confirmed"]  # sorted by name

    saved = json.loads((tmp_path / "01_explore_data" / "feature_selection_features_to_keep.json").read_text())
    assert saved == selected


def test_defensive_keeps_confirmed_and_tentative(feature_importance, tmp_path):
    selected = select_features(feature_importance, "boruta_style_defensive", save_path=str(tmp_path))
    assert selected == ["a_confirmed", "b_confirmed", "tentative"]


def test_max_features_keeps_highest_scores(feature_importance, tmp_path):
    selected = select_features(feature_importance, "boruta_style_defensive", max_features=2, save_path=str(tmp_path))
    assert selected == ["a_confirmed", "b_confirmed"]


def test_max_features_none_means_no_limit(feature_importance, tmp_path):
    selected = select_features(feature_importance, "boruta_style_defensive", max_features=None, save_path=str(tmp_path))
    assert len(selected) == 3


def test_unknown_strategy_raises(feature_importance, tmp_path):
    with pytest.raises(ValueError, match="Unknown feature_selection_strategy"):
        select_features(feature_importance, "kneedle", save_path=str(tmp_path))


def test_missing_columns_raise(tmp_path):
    with pytest.raises(ValueError, match="missing required columns"):
        select_features(pd.DataFrame({"score": [1.0]}), save_path=str(tmp_path))


def test_nothing_classified_raises(tmp_path):
    data = pd.DataFrame({"boruta_like_result": [None], "score": [1.0]}, index=["x"])
    with pytest.raises(ValueError, match="No classified features"):
        select_features(data, save_path=str(tmp_path))
