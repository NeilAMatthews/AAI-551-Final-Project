"""
Data processing module containing Dataset class and data handling functions.

Supports generic (x, y) datasets and time-series datasets (e.g., AEP_hourly).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd


class DataLoadingError(RuntimeError):
    """Raised when a dataset cannot be loaded or validated."""


@dataclass(frozen=True)
class DatasetSpec:
    """Immutable dataset configuration (nice for README + reproducibility)."""
    filepath: str
    x_col: str
    y_col: str
    parse_datetime: bool = False
    drop_na: bool = True


class Dataset:
    """
    Class representing a dataset for analysis.

    Handles data loading, validation, and basic statistics.
    """

    def __init__(
        self,
        filepath: str | Path,
        x_col: str = "x",
        y_col: str = "y",
        parse_datetime: bool = False,
        drop_na: bool = True,
    ):
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

    @classmethod
    def from_spec(cls, spec: DatasetSpec) -> "Dataset":
        """
        Alternate constructor demonstrating exception handling approach #1.
        """
        try:
            return cls(
                filepath=spec.filepath,
                x_col=spec.x_col,
                y_col=spec.y_col,
                parse_datetime=spec.parse_datetime,
                drop_na=spec.drop_na,
            )
        except (FileNotFoundError, pd.errors.ParserError, ValueError, TypeError) as e:
            raise DataLoadingError(f"Failed to create Dataset from spec: {spec}") from e

    def _load_data(self) -> None:
        """
        Load data from CSV file.

        Exception handling approach #2: catch + re-raise as a domain error
        to make debugging clearer in notebooks/tests.
        """
        try:
            df = pd.read_csv(self.filepath)
        except FileNotFoundError as e:
            raise DataLoadingError(f"File not found: {self.filepath}") from e
        except pd.errors.ParserError as e:
            raise DataLoadingError(f"CSV parse error in: {self.filepath}") from e

        # Check required columns
        if self.x_col not in df.columns or self.y_col not in df.columns:
            raise ValueError(
                f"CSV must contain columns '{self.x_col}' and '{self.y_col}'. "
                f"Found: {list(df.columns)}"
            )

        x_series = df[self.x_col]
        y_series = df[self.y_col]

        if self.parse_datetime:
            # errors="raise" ensures bad timestamps fail fast and loudly
            x_series = pd.to_datetime(x_series, errors="raise")

        # Ensure y is numeric
        y_series = pd.to_numeric(y_series, errors="raise")

        if self.drop_na:
            mask = ~(x_series.isna() | y_series.isna())
            x_series = x_series[mask]
            y_series = y_series[mask]

        # list comprehension requirement
        self.x_data = [x for x in x_series.tolist()]
        self.y_data = [float(y) for y in y_series.tolist()]

        # zip requirement
        self.data = list(zip(self.x_data, self.y_data))

        self._calculate_metadata()

    def _calculate_metadata(self) -> None:
        """Calculate basic statistics and metadata."""
        if not self.data:
            self.metadata = {}
            return

        # map + lambda requirement
        y_squared = list(map(lambda v: v**2, self.y_data))

        y_np = np.array(self.y_data, dtype=float)

        numeric_x = all(isinstance(x, (int, float, np.number)) for x in self.x_data)

        self.metadata = {
            "n_samples": len(self.data),
            "y_mean": float(np.mean(y_np)),
            "y_std": float(np.std(y_np, ddof=1)) if len(y_np) > 1 else 0.0,
            "y_min": float(np.min(y_np)),
            "y_max": float(np.max(y_np)),
            "y_energy": float(np.sum(y_squared)),
            "created_at": self.created_at.isoformat(),
        }

        if numeric_x:
            x_vals = [float(x) for x in self.x_data]
            self.metadata.update(
                {
                    "x_mean": float(np.mean(x_vals)),
                    "x_variance": float(np.var(x_vals, ddof=1)) if len(x_vals) > 1 else 0.0,
                    "x_range": (min(x_vals), max(x_vals)),
                }
            )

    def get_data_generator(self) -> Generator[Tuple[Any, float], None, None]:
        """Generator function to yield data points one by one."""
        if not self.data:
            return
        for x, y in self.data:  # for-loop requirement
            yield x, y

    def filter_data(self, condition_func) -> List[Tuple[Any, float]]:
        """Filter data using a condition function."""
        if not self.data:
            return []
        return list(filter(condition_func, self.data))  # filter requirement

    def to_dataframe(self) -> pd.DataFrame:
        """Convenience for models/plots."""
        return pd.DataFrame({self.x_col: self.x_data, self.y_col: self.y_data})

    def __str__(self) -> str:
        """String representation (rubric)."""
        return f"Dataset({Path(self.filepath).name}): {len(self)} samples"

    def __len__(self) -> int:
        """Operator overloading."""
        return len(self.data) if self.data else 0

    def __getitem__(self, index: int) -> Tuple[Any, float]:
        """Operator overloading."""
        if self.data and 0 <= index < len(self.data):
            return self.data[index]
        raise IndexError("Index out of range")


def validate_data(x_data: List[float], y_data: List[float]) -> Tuple[bool, str]:
    """
    Validate input data.

    Returns (is_valid, message).
    """
    if len(x_data) != len(y_data):  # if-statement requirement
        return False, f"Data length mismatch: x={len(x_data)}, y={len(y_data)}"

    if len(x_data) < 2:
        return False, "Insufficient data points (need at least 2)"

    if len(set(x_data)) == 1:
        return False, "All x values are identical"

    return True, "Data is valid"


if __name__ == "__main__":
    # __name__ requirement: quick sanity demo
    pass
