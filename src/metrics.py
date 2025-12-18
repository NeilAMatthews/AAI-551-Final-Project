"""
Metrics module for evaluating forecasting models.

Provides common error metrics for time series forecasting evaluation.
"""

from __future__ import annotations

from typing import Union

import numpy as np


def mae(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Mean absolute error
    
    Raises:
        ValueError: If arrays have different lengths or are empty
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Arrays must have same length: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    Calculate Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Mean squared error
    
    Raises:
        ValueError: If arrays have different lengths or are empty
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Arrays must have same length: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Root mean squared error
    
    Raises:
        ValueError: If arrays have different lengths or are empty
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: True values (must be non-zero)
        y_pred: Predicted values
    
    Returns:
        Mean absolute percentage error (as percentage, e.g., 15.5 for 15.5%)
    
    Raises:
        ValueError: If arrays have different lengths, are empty, or contain zeros
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Arrays must have same length: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    
    if np.any(y_true == 0):
        raise ValueError("y_true contains zero values, MAPE is undefined")
    
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def r2_score(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Arrays must have same length: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        # All y_true values are the same
        return 0.0 if ss_res == 0 else float('-inf')
    
    return float(1 - (ss_res / ss_tot))


if __name__ == "__main__":
    print(f"MAE: {mae(y_true, y_pred):.2f}")
    print(f"RMSE: {rmse(y_true, y_pred):.2f}")
    print(f"MAPE: {mape(y_true, y_pred):.2f}%")
    print(f"R²: {r2_score(y_true, y_pred):.4f}")
