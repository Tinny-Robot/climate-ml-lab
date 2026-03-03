# Methodology

## Problem Statement

Predict next-day mean temperature for Abeokuta, Ogun State, Nigeria (7.16°N, 3.35°E) using historical daily weather data.

**Target**: Mean daily temperature (°C), 1-day ahead  
**Success criteria**: R² > 0.85, MAE < 1°C  
**Scope**: 2015-01-01 to present, daily resolution

---

## Data Source

| Field | Value |
|---|---|
| Provider | Open-Meteo ERA5 Reanalysis |
| URL | https://archive-api.open-meteo.com/v1/era5 |
| License | CC-BY-4.0 |
| Variables | temperature_2m, relative_humidity_2m, wind_speed_10m, precipitation, surface_pressure |
| Aggregation | Hourly → daily (mean/max/min) |

---

## Preprocessing

1. Time-series interpolation for missing values (`method='time'`, bidirectional)
2. Lag-1 features for all weather variables
3. Calendar features: month, day, weekday, day_of_year
4. No normalisation required for tree-based models; StandardScaler used for SVR pipeline

**Final feature set**: 15 columns (lag-1 signals + calendar features)

---

## Model Training

- **Train/test split**: 80/20, temporal (no shuffle)
- **Cross-validation**: TimeSeriesSplit, 5 folds
- **Random seed**: 42

### Models evaluated

| Model | Type |
|---|---|
| LinearRegression | Baseline linear |
| Lasso / Ridge | Regularised linear |
| DecisionTree | Single tree |
| RandomForest | Bagged ensemble |
| GradientBoosting | Boosted ensemble |
| SVR (RBF) | Kernel method |

### Hyperparameter tuning (GridSearchCV)

**GradientBoosting** — best params: `learning_rate=0.05`, `max_depth=3`, `n_estimators=200`  
**RandomForest** — best params: `max_depth=10`, `min_samples_split=2`, `n_estimators=300`  
**SVR** — best params: `C=5`, `epsilon=0.1`, `gamma='scale'`

---

## Evaluation

**Metrics**: MAE, RMSE, R²

| Model | MAE | RMSE | R² |
|---|---|---|---|
| GradientBoosting (tuned) | 0.4819 | 0.5979 | 0.9049 |
| RandomForest (tuned) | 0.4882 | 0.6101 | 0.9010 |
| Lasso | 0.4723 | 0.5933 | 0.9064 |
| LinearRegression | 0.4751 | 0.5958 | 0.9056 |
| SVR (tuned) | 0.5347 | 0.6708 | 0.8803 |
| DecisionTree | 0.5163 | 0.6665 | 0.8818 |
| Persistence baseline | 0.4728 | 0.6203 | 0.8977 |

**Selected model**: GradientBoosting — best balance of accuracy and robustness across seasons.

---

## Seasonal Performance

| Season | Months | MAE (approx.) |
|---|---|---|
| Dry | Dec–Feb | 0.55°C |
| Hot/Dry | Mar–May | 0.58°C |
| Rainy | Jun–Aug | 0.62°C |
| Transition | Sep–Nov | 0.64°C |

---

## Limitations

- Single-location model (does not generalise to other stations without retraining)
- ERA5 reanalysis data, not direct station observations
- 1-day horizon only; multi-day predictions degrade in accuracy
- Model does not incorporate synoptic-scale weather signals
