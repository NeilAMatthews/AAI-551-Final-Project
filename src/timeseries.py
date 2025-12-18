from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_processing import Dataset


@dataclass(frozen=True)
class SplitSpec:
    """Immutable split config."""
    test_size: float = 0.2


class TimeSeriesDataset(Dataset):
    """
    Dataset subclass for time-series.

    Inheritance relationship: TimeSeriesDataset IS-A Dataset.
    """

    def __init__(
        self,
        filepath: str,
        x_col: str = "Datetime",
        y_col: str = "AEP_MW",
        drop_na: bool = True,
    ):
        super().__init__(
            filepath=filepath,
            x_col=x_col,
            y_col=y_col,
            parse_datetime=True,
            drop_na=drop_na,
        )
        self.data_type = "time_series"

        # Ensure sorted by time
        df = self.to_dataframe().sort_values(self.x_col)
        self.x_data = df[self.x_col].tolist()
        self.y_data = df[self.y_col].astype(float).tolist()
        self.data = list(zip(self.x_data, self.y_data))

    def slice_range(self, start: Optional[str], end: Optional[str]) -> "TimeSeriesDataset":
        """
        Return a new TimeSeriesDataset-like object containing only [start, end].
        """
        df = self.to_dataframe()
        if start is not None:
            start_dt = pd.to_datetime(start)
            df = df[df[self.x_col] >= start_dt]
        if end is not None:
            end_dt = pd.to_datetime(end)
            df = df[df[self.x_col] <= end_dt]

        # Create a lightweight clone without re-reading CSV
        clone = object.__new__(TimeSeriesDataset)
        Dataset.__init__(clone, self.filepath, self.x_col, self.y_col, True, self.drop_na)  # type: ignore
        clone.x_data = df[self.x_col].tolist()
        clone.y_data = df[self.y_col].astype(float).tolist()
        clone.data = list(zip(clone.x_data, clone.y_data))
        clone.data_type = "time_series"
        clone._calculate_metadata()
        return clone

    def add_calendar_features(self) -> pd.DataFrame:
        """
        Build a supervised learning table with calendar-based features.
        """
        df = self.to_dataframe().copy()
        dt = pd.to_datetime(df[self.x_col])

        # list comprehension requirement (again, but harmless)
        df["hour"] = [t.hour for t in dt]
        df["dayofweek"] = [t.dayofweek for t in dt]
        df["month"] = [t.month for t in dt]

        return df

    def make_lag_features(self, lags: List[int]) -> pd.DataFrame:
        """
        Create lag features for forecasting.

        Example: lags=[1, 24, 168] -> previous hour, previous day, previous week.
        """
        df = self.add_calendar_features()
        df = df.sort_values(self.x_col).reset_index(drop=True)

        for lag in lags:
            df[f"lag_{lag}"] = df[self.y_col].shift(lag)

        df = df.dropna().reset_index(drop=True)
        return df

    def train_test_split(self, split: SplitSpec = SplitSpec()) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Time-ordered split (no shuffling).
        """
        df = self.to_dataframe().sort_values(self.x_col).reset_index(drop=True)
        n = len(df)
        cut = int(n * (1.0 - split.test_size))
        return df.iloc[:cut].copy(), df.iloc[cut:].copy()
