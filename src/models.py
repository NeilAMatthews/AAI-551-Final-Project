from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import mae


@dataclass
class FitResult:
    """Mutable result container (dataclass is fine as mutable)."""
    model_name: str
    params: Dict[str, float]
    train_mae: float


class BaseForecaster:
    """Simple base class for forecasters."""

    def fit(self, series: np.ndarray) -> FitResult:
        raise NotImplementedError

    def predict(self, steps: int) -> np.ndarray:
        raise NotImplementedError


class NaiveLastValueForecaster(BaseForecaster):
    """Predicts the last observed value forward."""

    def __init__(self):
        self.last_value: Optional[float] = None

    def fit(self, series: np.ndarray) -> FitResult:
        if len(series) == 0:
            raise ValueError("Empty series")
        self.last_value = float(series[-1])
        preds = np.full_like(series, self.last_value, dtype=float)
        return FitResult("naive_last", {}, float(mae(series, preds)))

    def predict(self, steps: int) -> np.ndarray:
        if self.last_value is None:
            raise RuntimeError("Model not fit")
        return np.full(steps, self.last_value, dtype=float)


class MovingAverageForecaster(BaseForecaster):
    """Predicts with the rolling mean of the last `window` values."""

    def __init__(self, window: int = 24):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.history: Optional[np.ndarray] = None

    def fit(self, series: np.ndarray) -> FitResult:
        if len(series) < self.window:
            raise ValueError("Series shorter than window")
        self.history = series.astype(float)

        # In-sample “prediction”: rolling mean (simple baseline)
        preds = np.zeros_like(series, dtype=float)
        preds[: self.window] = np.mean(series[: self.window])
        for i in range(self.window, len(series)):  # for-loop requirement (again)
            preds[i] = float(np.mean(series[i - self.window : i]))
        return FitResult("moving_avg", {"window": float(self.window)}, float(mae(series, preds)))

    def predict(self, steps: int) -> np.ndarray:
        if self.history is None:
            raise RuntimeError("Model not fit")
        hist = self.history.copy()
        out = []
        for _ in range(steps):
            out.append(float(np.mean(hist[-self.window :])))
            hist = np.append(hist, out[-1])
        return np.array(out, dtype=float)


def tune_moving_average_window(
    train_series: np.ndarray,
    start: int = 6,
    stop: int = 200,
    step: int = 6,
) -> Tuple[MovingAverageForecaster, FitResult]:
    """
    Uses a WHILE LOOP to tune the moving average window size.

    Returns the best model + its fit result.
    """
    best_model: Optional[MovingAverageForecaster] = None
    best_result: Optional[FitResult] = None

    w = start
    while w <= stop:  # while-loop requirement
        model = MovingAverageForecaster(window=w)
        result = model.fit(train_series)

        if best_result is None or result.train_mae < best_result.train_mae:  # if-statement (again)
            best_model, best_result = model, result

        w += step

    assert best_model is not None and best_result is not None
    return best_model, best_result


class LinearRegressionForecaster:
    """
    Simple linear regression forecaster using NumPy least squares.
    Uses calendar + lag features created in TimeSeriesDataset.
    """

    def __init__(self):
        self.coef_: Optional[np.ndarray] = None

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str], y_col: str) -> FitResult:
        X = train_df[feature_cols].to_numpy(dtype=float)
        y = train_df[y_col].to_numpy(dtype=float)

        # Add bias term
        Xb = np.column_stack([np.ones(len(X)), X])
        self.coef_ = np.linalg.lstsq(Xb, y, rcond=None)[0]

        preds = self.predict(train_df, feature_cols)
        return FitResult("linreg", {}, float(mae(y, preds)))

    def predict(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model not fit")
        X = df[feature_cols].to_numpy(dtype=float)
        Xb = np.column_stack([np.ones(len(X)), X])
        return (Xb @ self.coef_).astype(float)
