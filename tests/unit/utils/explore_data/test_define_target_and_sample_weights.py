import pandas as pd
import pytest

from power_quant_strategies.utils.explore_data.define_target_and_sample_weights import define_target_and_sample_weights


def test_target_is_exit_above_entry_and_weight_is_absolute_spread():
    data = pd.DataFrame({"da_price": [50.0, 80.0, 30.0], "imb_price": [70.0, 60.0, 30.0], "load": [1, 2, 3]})

    result = define_target_and_sample_weights(data, column_name_entry="da_price", column_name_exit="imb_price")

    assert result["target"].tolist() == [True, False, False]
    assert result["sample_weight"].tolist() == [20.0, 20.0, 0.0]
    # the prices would leak the target, so they are dropped; other features stay
    assert list(result.columns) == ["load", "sample_weight", "target"]
    # the input is not changed
    assert list(data.columns) == ["da_price", "imb_price", "load"]


def test_custom_target_column_name():
    data = pd.DataFrame({"entry": [1.0], "exit": [2.0]})
    result = define_target_and_sample_weights(data, column_name_entry="entry", column_name_exit="exit", column_name_target="y")
    assert result["y"].tolist() == [True]


@pytest.mark.parametrize(("entry", "exit"), [("missing", "imb_price"), ("da_price", "missing")])
def test_missing_price_column_raises(entry, exit):
    data = pd.DataFrame({"da_price": [1.0], "imb_price": [2.0]})
    with pytest.raises(AssertionError):
        define_target_and_sample_weights(data, column_name_entry=entry, column_name_exit=exit)
