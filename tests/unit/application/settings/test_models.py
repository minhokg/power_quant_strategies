import pytest
from pydantic import ValidationError

from power_quant_strategies.application.settings.models import BaseSettings


def test_defaults_match_the_pipeline():
    settings = BaseSettings()
    assert settings.column_name_entry == "da_price"
    assert settings.column_name_exit == "imb_price"
    assert settings.column_name_target == "target"
    assert settings.percentage_train_hyperparameter_tuning + settings.percentage_valid_hyperparameter_tuning < 1  # leaves a test split
    assert settings.feature_selection_strategy in {"boruta_style_aggressive", "boruta_style_defensive"}


def test_values_can_be_overridden_and_are_validated():
    assert BaseSettings(n_trials="5").n_trials == 5
    with pytest.raises(ValidationError):
        BaseSettings(n_trials="many")
