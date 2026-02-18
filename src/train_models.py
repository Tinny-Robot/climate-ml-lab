"""
Model training utilities for climate ML lab.

Provides functions to:
- Load processed data
- Build features/target for next-day temperature forecasting
- Train multiple regression models
- Evaluate (MAE, RMSE, R2)
- Select and save the best model
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .utils import PROCESSED_DIR, REPORTS_DIR, SAVED_MODELS_DIR, get_logger

LOGGER = get_logger(__name__)


@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]


def load_processed(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        path = PROCESSED_DIR / "clean_weather.csv"
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    return df


def build_supervised(
    df: pd.DataFrame,
    lags: Tuple[int, ...] | List[int] = (1,),
    rolling_windows: Tuple[int, ...] | List[int] = (),
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create features and target for next-day temperature.

    Target: temperature_c shifted by -1 day (next day's temp).
    Features: all lag-1 features and calendar features.
    """
    if "temperature_c" not in df.columns:
        raise ValueError("Expected 'temperature_c' in processed data")

    target = df["temperature_c"].shift(-1).rename("target_temp_next_day")

    # Base features: existing lag-1 and time features (from preprocessing)
    feature_cols = [c for c in df.columns if c.endswith("_lag1") or c in ["month", "day", "weekday"]]

    # Optionally add additional lags from original signals
    base_signals = [
        "temperature_c",
        "humidity_pct",
        "wind_speed_mps",
        "precipitation_mm",
        "surface_pressure_hpa",
    ]
    for L in lags:
        if L == 1:
            continue  # already included from preprocessing
        for sig in base_signals:
            col = f"{sig}_lag{L}"
            if col not in df:
                df[col] = df[sig].shift(L)
            feature_cols.append(col)

    # Optionally add rolling means from original signals
    for W in rolling_windows:
        for sig in base_signals:
            col = f"{sig}_roll{W}"
            if col not in df:
                df[col] = df[sig].rolling(W, min_periods=max(1, W // 2)).mean()
            feature_cols.append(col)

    # Build feature frame
    X = df[sorted(set(feature_cols))]

    # Align and drop rows where target is NaN (last row after shift)
    data = pd.concat([X, target], axis=1).dropna()
    X = data[feature_cols]
    y = data["target_temp_next_day"]
    return X, y


def split_dataset(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Dataset:
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=test_size, shuffle=False
    )
    return Dataset(X_train, X_test, y_train, y_test, feature_names=list(X.columns))


def get_models() -> Dict[str, object]:
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        # Baseline SVR with scaling pipeline (helps stability)
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.1))
        ]),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def tune_svr(dataset: Dataset) -> Tuple[object, Dict[str, object], float]:
    """Hyperparameter tuning for SVR using TimeSeriesSplit CV with a scaling pipeline."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf"))
    ])

    param_grid = {
        "svr__C": [1.0, 5.0, 10.0, 50.0, 100.0],
        "svr__epsilon": [0.01, 0.05, 0.1, 0.2],
        "svr__gamma": ["scale", "auto"],
    }

    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
        verbose=0,
    )

    grid.fit(dataset.X_train, dataset.y_train)
    best_est = grid.best_estimator_
    best_params = grid.best_params_
    best_score = float(-grid.best_score_)
    LOGGER.info("SVR tuning best RMSE (CV): %.3f with params %s", best_score, best_params)
    return best_est, best_params, best_score


def train_and_evaluate(dataset: Dataset, enable_svr_tuning: bool = True) -> Tuple[Dict[str, Dict[str, float]], Tuple[str, object]]:
    models = get_models()
    metrics: Dict[str, Dict[str, float]] = {}

    best_name = None
    best_model = None
    best_rmse = float("inf")

    for name, model in models.items():
        LOGGER.info("Training %s", name)
        model.fit(dataset.X_train, dataset.y_train)
        preds = model.predict(dataset.X_test)
        m = evaluate(dataset.y_test, preds)
        metrics[name] = m
        LOGGER.info("%s -> MAE=%.3f RMSE=%.3f R2=%.3f", name, m["MAE"], m["RMSE"], m["R2"])
        if m["RMSE"] < best_rmse:
            best_rmse = m["RMSE"]
            best_name = name
            best_model = model

    # Optional: hyperparameter-tuned SVR
    if enable_svr_tuning:
        LOGGER.info("Starting SVR hyperparameter tuning with TimeSeriesSplit...")
        tuned_model, best_params, cv_rmse = tune_svr(dataset)
        preds = tuned_model.predict(dataset.X_test)
        m = evaluate(dataset.y_test, preds)
        metrics["SVR_Tuned"] = {**m, "CV_RMSE": cv_rmse, "best_params": best_params}
        LOGGER.info("SVR_Tuned -> MAE=%.3f RMSE=%.3f R2=%.3f (CV_RMSE=%.3f)", m["MAE"], m["RMSE"], m["R2"], cv_rmse)
        if m["RMSE"] < best_rmse:
            best_rmse = m["RMSE"]
            best_name = "SVR_Tuned"
            best_model = tuned_model

    assert best_name is not None and best_model is not None
    return metrics, (best_name, best_model)


def save_best_model(model: object, name: str, out_dir: Path | None = None) -> Path:
    if out_dir is None:
        out_dir = SAVED_MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "best_model.joblib"
    joblib.dump({"model": model, "name": name}, path)
    LOGGER.info("Saved best model (%s) to %s", name, path)
    return path


__all__ = [
    "Dataset",
    "load_processed",
    "build_supervised",
    "split_dataset",
    "get_models",
    "evaluate",
    "train_and_evaluate",
    "save_best_model",
    "save_metrics",
]


def save_metrics(metrics: dict, out_dir: Path | None = None) -> Path:
    if out_dir is None:
        out_dir = REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "evaluation_metrics.json"

    # Convert non-serializable items
    for model, m in metrics.items():
        for k, v in m.items():
            if isinstance(v, (np.generic, np.ndarray)):
                metrics[model][k] = v.item()
            elif k == 'best_params' and isinstance(v, dict):
                 for p_k, p_v in v.items():
                     if isinstance(p_v, (np.generic, np.ndarray)):
                         v[p_k] = p_v.item()


    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    LOGGER.info("Saved evaluation metrics to %s", path)
    return path


def main():
    """Main entry point to run the training pipeline."""
    LOGGER.info("Starting model training pipeline...")
    df = load_processed()
    X, y = build_supervised(df)
    dataset = split_dataset(X, y)
    metrics, (best_name, best_model) = train_and_evaluate(dataset)
    save_best_model(best_model, best_name)
    save_metrics(metrics)
    LOGGER.info("Model training pipeline finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
