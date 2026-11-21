# power_quant_strategies

Quantitative trading strategies for electricity markets using machine learning, optimization, and statistical modeling.

The main strategy trades the spread between the **day-ahead price** (entry) and the **imbalance price** (exit) in the
German/Luxembourg bidding zone (`DE_LU`). A classifier predicts whether the imbalance price will end up above the
day-ahead price. Its score decides the direction (long above an optimal threshold, short below), and the score's
distance from the threshold sets the bet size. Strategies are judged by PnL, Sharpe ratio, drawdown and classification
metrics in a walk-forward backtest.

## Pipeline

```
load data ──► explore ──► select features ──► tune hyperparameters ──► walk-forward backtest ──► evaluate
ENTSO-E,      stationarity,  boruta-like      Optuna + XGBoost          daily re-fit on a          trading +
Open-Meteo    autocorr.,     importance                                 183-day sliding window     classification
              distributions                                                                         metrics, PnL plots
```

1. **Load**: `load_data_from_entsoe` (load / wind & solar / generation forecasts, day-ahead and imbalance prices) and
   `load_data_from_open_meteo` (hourly weather).
2. **Target**: `define_target_and_sample_weights` sets `target = exit > entry` and `sample_weight = |exit − entry|`.
3. **Explore**: `check_stationarity` (ADF, KPSS, Phillips-Perron), `check_autocorrelation` (ACF/PACF, Ljung-Box, runs
   test), `find_best_parametric_distribution_fit`.
4. **Feature selection**: `calculate_feature_importance` compares real features with shuffled "fake" copies across
   seeds and time-series folds; `select_features` keeps the Confirmed (and, in defensive mode, Tentative) features.
5. **Tune**: `tune_hyperparameter_xgboost` runs an Optuna search over XGBoost parameters.
6. **Backtest**: `fit_xgboost_walk_forward` trains on a sliding window of days and tests on the next day, for every day.
7. **Evaluate**: `find_optimal_threshold_and_maximum_pnl`, `evaluate_trading_metrics`,
   `evaluate_classification_metrics`, `evaluate_trading_plots`.

The `application/mains/` functions run steps 4–7 end to end with the settings in `BaseSettings`.

## Project structure

```
src/power_quant_strategies/
├── application/
│   ├── mains/            # entry points: feature selection, hyperparameter tuning, walk-forward training
│   └── settings/         # BaseSettings (pydantic): column names, split sizes, Optuna and XGBoost settings
├── models/               # ARIMA and LSTM model definitions
└── utils/
    ├── load_data/        # ENTSO-E client and loader, Open-Meteo loader
    ├── explore_data/     # target definition, statistical tests, feature importance
    ├── preprocess_data/  # feature selection
    ├── fit_model/        # XGBoost fit / tuning / walk forward, spread and threshold helpers
    ├── evaluate_model/   # classification and trading metrics, plots
    └── helper/           # logging, NaN removal, path shortening, CPU count, snake case
tests/unit/               # mirrors src/power_quant_strategies/
```

Results are written under `settings.save_path` in numbered folders: `01_explore_data`, `02_fit_model`,
`03_evaluate_model`.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. `uv sync` creates the virtual environment and installs all dependencies (Python ≥ 3.11).
3. `uv run pre-commit install` sets up the git hooks (ruff lint and format).
4. ENTSO-E data needs an API key in the environment variable `ENTSOE_API_KEY`
   ([how to get one](https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html#_authentication_and_authorisation)).
   Open-Meteo needs no key.

## Usage

```python
import pandas as pd

from power_quant_strategies.application.mains.main_feature_importance_and_selection import main_feature_importance_and_selection
from power_quant_strategies.application.mains.main_train_walk_forward import main_train_walk_forward
from power_quant_strategies.application.mains.main_tune_hyperparameter import main_tune_hyperparameter
from power_quant_strategies.application.settings.models import BaseSettings
from power_quant_strategies.utils.explore_data.define_target_and_sample_weights import define_target_and_sample_weights
from power_quant_strategies.utils.load_data.load_data_from_entsoe import load_data_from_entsoe

settings = BaseSettings(save_path="results")

data = load_data_from_entsoe(start_date=pd.Timestamp("2025-01-01"), end_date=pd.Timestamp("2026-01-01"))
data = define_target_and_sample_weights(data, column_name_entry=settings.column_name_entry, column_name_exit=settings.column_name_exit)

features = main_feature_importance_and_selection(data=data, settings=settings)
data = data[features + [settings.column_name_target, "sample_weight"]]

hyperparameter = main_tune_hyperparameter(data=data, settings=settings)
main_train_walk_forward(data=data, settings=settings, hyperparameter=hyperparameter)
```

## Development

```bash
uv run pytest tests/unit                 # unit tests
uv run pytest tests/unit --cov=src/power_quant_strategies --cov-report=term-missing
uvx ruff@0.8.6 check .                   # lint (same version as pre-commit and CI)
uvx ruff@0.8.6 format --check .          # format
```

CI (`.github/workflows/cicd.yml`) runs ruff and the unit tests on every pull request to `main` and posts a coverage
comment.

Conventions: one module per public function (`utils/<area>/<function_name>.py`), reST docstrings with `:param:` for
every argument, ruff line length 200. Branches are named `<issue>-<title>`, and commits `feat #<issue>: <summary>`.

## License

MIT, see [LICENSE](LICENSE).
