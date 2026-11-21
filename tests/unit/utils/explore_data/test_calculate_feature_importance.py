import numpy as np
import pandas as pd
import pytest

from power_quant_strategies.utils.explore_data import calculate_feature_importance as module
from power_quant_strategies.utils.explore_data.calculate_feature_importance import _create_fake_features, _parse_xgb_tree_features, calculate_feature_importance


@pytest.fixture(autouse=True)
def single_cpu(monkeypatch):
    monkeypatch.setattr(module, "get_usable_cpu_count", lambda: 1)


@pytest.fixture
def data() -> pd.DataFrame:
    """Return data whose target depends on "signal" only; "noise" and "empty" carry no information."""
    rng = np.random.default_rng(0)
    n = 300
    signal = rng.normal(size=n)
    return pd.DataFrame(
        {
            "target": (signal + rng.normal(scale=0.3, size=n) > 0).astype(int),
            "sample_weight": rng.uniform(0.5, 2, n),
            "signal": signal,
            "noise": rng.normal(size=n),
            "empty": np.nan,
        },
        index=pd.date_range("2026-01-01", periods=n, freq="h"),
    )


def test_create_fake_features_permutes_values():
    real = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0) * 2})
    fake = _create_fake_features(real, ["a"], fake_feature_prefix="fake_", random_state=0)
    assert list(fake.columns) == ["fake_a"]
    assert sorted(fake["fake_a"]) == list(real["a"])
    assert not fake["fake_a"].equals(real["a"])
    pd.testing.assert_index_equal(fake.index, real.index)


def test_parse_xgb_tree_features_counts_trees_per_feature():
    trees = pd.DataFrame({"Tree": [0, 0, 0, 1, 1], "Feature": ["a", "a", "Leaf", "b", "a"]})
    counts = _parse_xgb_tree_features(trees)
    assert dict(counts) == {"a": 2, "b": 1}


def test_signal_ranks_first_and_unusable_features_are_rejected(data, tmp_path):
    result = calculate_feature_importance(data, n_splits=2, n_seeds=1, save_path=str(tmp_path))

    assert result.index[0] == "signal"
    assert result.loc["signal", "rank"] == 1
    assert result.loc["signal", "boruta_like_result"] == "Confirmed"
    assert result.loc["empty", "boruta_like_result"] == "Rejected"
    assert pd.isna(result.loc["empty", "rank"])
    assert set(result.index) == {"signal", "noise", "empty"}  # no fake features in the output
    assert (tmp_path / "01_explore_data" / "feature_importance.parquet").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.drop(columns="target"), "target column missing"),
        (lambda d: d.drop(columns="sample_weight"), "sample_weight column missing"),
        (lambda d: d.assign(target=2), "target must be binary"),
        (lambda d: d.assign(sample_weight=-1.0), "invalid sample_weight"),
        (lambda d: d[["target", "sample_weight"]], "no features"),
        (lambda d: d[["target", "sample_weight", "empty"]], "no usable features"),
        (lambda d: d.rename(columns={"noise": "fake_noise"}), "collide"),
    ],
)
def test_invalid_input_raises(data, change, message):
    with pytest.raises(ValueError, match=message):
        calculate_feature_importance(change(data), n_splits=2, n_seeds=1, save_path=None)
