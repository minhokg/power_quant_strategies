from pydantic import BaseModel


class BaseSettings(BaseModel):
    """Model including all base settings."""

    percentage_train_hyperparameter_tuning = 0.6
    percentage_valid_hyperparameter_tuning = 0.2
    column_name_target = "target"
    n_estimators = 500
    early_stopping_rounds = 10
    verbose = False
    direction = "maximize"
    random_state = 42
    n_trials = 100
    max_run_time_per_model_fit = 20
    show_progress_bar = True
    shuffle = True
    device = "cpu"
