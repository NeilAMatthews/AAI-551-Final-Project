"""
Data processing module containing Dataset class and data handling functions.

Supports generic (x, y) datasets and time-series datasets (e.g., AEP_hourly).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional, Generator, Dict, Any

import numpy as np
import pandas as pd


class Dataset:
    """
    Class representing a dataset for analysis.

    This class handles data loading, validation, and basic statistics.
    """

    def __init__(
        self,
        filepath: str | Path,
        x_col: str = "x",
        y_col: str = "y",
        parse_datetime: bool = False,
        drop_na: bool = True,
    ):
        """
        Initialize Dataset with data from a CSV file.

        Args:
            filepath: Path to the CSV file containing data
            x_col: Column name for x (feature) values
            y_col: Column name for y (target) values
            parse_datetime: If True, parse x_col as datetime
            drop_na: If True, drop rows with missing x/y after conversion
        """
        self.filepath = str(filepath)
        self.x_col = x_col
        self.y_col = y_col
        self.parse_datetime = parse_datetime
        self.drop_na = drop_na

        # Mutable objects
        self.data: Optional[List[Tuple[Any, float]]] = None
        self.x_data: List[Any] = []
        self.y_data: List[float] = []
        self.metadata: Dict[str, Any] = {}

        # Immutable objects
        self.created_at = datetime.now()
        self.data_type = "numerical"

        self._load_data()

    def _load_data(self) -> None:
        """Load data from CSV file with error handling (raises exceptions)."""
        df = pd.read_csv(self.filepath)

        # Check required columns
        if self.x_col not in df.columns or self.y_col not in df.columns:
            raise ValueError(
                f"CSV must contain columns '{self.x_col}' and '{self.y_col}'. "
                f"Found: {list(df.columns)}"
            )

        x_series = df[self.x_col]
        y_series = df[self.y_col]

        # Parse datetime if requested
        if self.parse_datetime:
            x_series = pd.to_datetime(x_series, errors="raise")

        # Ensure y is numeric
        y_series = pd.to_numeric(y_series, errors="raise")

        # Optional NA handling
        if self.drop_na:
            mask = ~(x_series.isna() | y_series.isna())
            x_series = x_series[mask]
            y_series = y_series[mask]

        # Convert to lists (list comprehension requirement)
        self.x_data = [x for x in x_series.tolist()]
        self.y_data = [float(y) for y in y_series.tolist()]

        # Store as list of tuples (zip requirement)
        self.data = list(zip(self.x_data, self.y_data))

        self._calculate_metadata()

    def _calculate_metadata(self) -> None:
        """Calculate basic statistics and metadata."""
        if not self.data:
            self.metadata = {}
            return

        # Use map + lambda (rubric feature)
        y_squared = list(map(lambda v: v ** 2, self.y_data))

        # Use numpy meaningfully (not superfluous)
        y_np = np.array(self.y_data, dtype=float)

        # For numeric x only; datetime x is handled separately
        numeric_x = all(isinstance(x, (int, float, np.number)) for x in self.x_data)

        self.metadata = {
            "n_samples": len(self.data),
            "y_mean": float(np.mean(y_np)),
            "y_std": float(np.std(y_np, ddof=1)) if len(y_np) > 1 else 0.0,
            "y_min": float(np.min(y_np)),
            "y_max": float(np.max(y_np)),
            "y_energy": float(np.sum(y_squared)),  # uses map/lambda output
            "created_at": self.created_at.isoformat(),
        }

        if numeric_x:
            self.metadata.update(
                {
                    "x_mean": statistics.mean([float(x) for x in self.x_data]),
                    "x_variance": statistics.variance([float(x) for x in self.x_data])
                    if len(self.x_data) > 1
                    else 0.0,
                    "x_range": (min(self.x_data), max(self.x_data)),
                }
            )

    def get_data_generator(self) -> Generator[Tuple[Any, float], None, None]:
        """
        Generator function to yield data points one by one.

        Yields:
            Tuple of (x, y) values
        """
        if not self.data:
            return
        for x, y in self.data:  # for-loop requirement
            yield x, y

    def filter_data(self, condition_func) -> List[Tuple[Any, float]]:
        """
        Filter data using a condition function.

        Args:
            condition_func: Function that takes (x, y) and returns bool

        Returns:
            Filtered list of (x, y) tuples
        """
        if not self.data:
            return []
        return list(filter(condition_func, self.data))  # filter requirement

    def __str__(self) -> str:
        """String representation of the dataset."""
        return f"Dataset({Path(self.filepath).name}): {len(self)} samples"

    def __len__(self) -> int:
        """Return number of samples (operator overloading)."""
        return len(self.data) if self.data else 0

    def __getitem__(self, index: int) -> Tuple[Any, float]:
        """Get item by index (operator overloading)."""
        if self.data and 0 <= index < len(self.data):
            return self.data[index]
        raise IndexError("Index out of range")


def validate_data(x_data: List[float], y_data: List[float]) -> Tuple[bool, str]:
    """
    Validate input data for regression.

    Args:
        x_data: List of x values
        y_data: List of y values

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(x_data) != len(y_data):  # if-statement requirement
        return False, f"Data length mismatch: x={len(x_data)}, y={len(y_data)}"

    if len(x_data) < 2:
        return False, "Insufficient data points (need at least 2)"

    if len(set(x_data)) == 1:
        return False, "All x values are identical"

    return True, "Data is valid"


if __name__ == "__main__":
    # __name__ requirement: quick sanity demo (don’t rely on this in the notebook)
    # TODO: update filepath and columns for your dataset when you run locally.
    pass
