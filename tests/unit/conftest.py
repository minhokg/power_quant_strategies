import matplotlib
import numpy as np
import pandas as pd
import pytest

# Plots are only saved to disk in the tests, never shown.
matplotlib.use("Agg")


@pytest.fixture
def trading_data() -> pd.DataFrame:
    """
    Hourly data over four days with a target, sample weights and model scores.

    The scores are informative: a higher ``preds`` means the spread is more likely positive.

    :return: DataFrame with columns ``target``, ``sample_weight``, ``preds`` and two features.
    """
    rng = np.random.default_rng(0)
    index = pd.date_range("2026-01-01", periods=96, freq="h", tz="Europe/Berlin")
    preds = rng.uniform(0, 1, len(index))
    target = (preds + rng.normal(0, 0.2, len(index)) > 0.5).astype(int)
    return pd.DataFrame(
        {
            "target": target,
            "sample_weight": rng.uniform(1, 10, len(index)),
            "preds": preds,
            "feature_a": rng.normal(size=len(index)),
            "feature_b": rng.normal(size=len(index)),
        },
        index=index,
    )


@pytest.fixture
def results_dir(tmp_path):
    """
    Temporary save_path with the numbered result folders the pipeline writes into.

    :param tmp_path: pytest's temporary directory.
    :return: Path of the save_path as str.
    """
    for folder in ("01_explore_data", "02_fit_model", "03_evaluate_model"):
        (tmp_path / folder).mkdir()
    return str(tmp_path)
