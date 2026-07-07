from pydantic import BaseModel


class BaseSettings(BaseModel):
    """Model including all base settings."""

    percentage_train_hyperparameter_tuning: float = 0.6
    percentage_valid_hyperparameter_tuning: float = 0.2
    column_name_target: str = "target"
    n_estimators: int = 500
    early_stopping_rounds: int = 10
    verbose: bool = False
    direction: str = "maximize"
    random_state: int = 42
    n_trials: int = 100
    max_run_time_per_model_fit: int = 20
    show_progress_bar: bool = True
    shuffle: bool = True
    device: str = "cpu"
