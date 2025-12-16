"""
Data processing module containing Dataset class and data handling functions.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Generator, Dict, Any
from datetime import datetime
import sys
import statistics
from pathlib import Path


class Dataset:
    """
    Class representing a dataset for linear regression analysis.
    
    This class handles data loading, validation, and basic statistics.
    """
    
    def __init__(self, filepath: str):
        """
        Initialize Dataset with data from a file.
        
        Args:
            filepath: Path to the CSV file containing data
        """
        self.filepath = filepath
        self.data = None  # Mutable list object
        self.x_data = []  # Mutable list object
        self.y_data = []  # Mutable list object
        self.metadata = {}  # Mutable dict object
        
        # Immutable objects
        self.created_at = datetime.now()
        self.data_type = "numerical"  # Immutable string
        
        self._load_data()
    
    def _load_data(self) -> None:
        """Load data from CSV file with error handling."""
        try:
            # Using pandas for data I/O
            df = pd.read_csv(self.filepath)
            
            # Check required columns
            if 'x' not in df.columns or 'y' not in df.columns:
                raise ValueError("CSV must contain 'x' and 'y' columns")
            
            # Convert to lists using list comprehension
            self.x_data = [float(x) for x in df['x'].tolist()]
            self.y_data = [float(y) for y in df['y'].tolist()]
            
            # Store as list of tuples
            self.data = list(zip(self.x_data, self.y_data))
            
            # Calculate metadata
            self._calculate_metadata()
            
        except FileNotFoundError:
            print(f"Error: File '{self.filepath}' not found.")
            sys.exit(1)
        except pd.errors.EmptyDataError:
            print("Error: CSV file is empty.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            sys.exit(1)
    
    def _calculate_metadata(self) -> None:
        """Calculate basic statistics and metadata."""
        if not self.data:
            return
        
        # Using map and lambda for calculations
        x_squared = list(map(lambda x: x**2, self.x_data))
        y_squared = list(map(lambda y: y**2, self.y_data))
        
        self.metadata = {
            'n_samples': len(self.data),
            'x_mean': statistics.mean(self.x_data),
            'y_mean': statistics.mean(self.y_data),
            'x_variance': statistics.variance(self.x_data),
            'y_variance': statistics.variance(self.y_data),
            'x_range': (min(self.x_data), max(self.x_data)),
            'y_range': (min(self.y_data), max(self.y_data))
        }
    
    def get_data_generator(self) -> Generator[Tuple[float, float], None, None]:
        """
        Generator function to yield data points one by one.
        
        Yields:
          Tuple of (x, y) values
        """
        for x, y in self.data:
            yield x, y
    
    def filter_data(self, condition_func) -> List[Tuple[float, float]]:
        """
        Filter data using a condition function.
        
        Args:
            condition_func: Function that takes (x, y) and returns bool
            
        Returns:
            Filtered list of (x, y) tuples
        """
        return list(filter(condition_func, self.data))
    
    def get_subset(self, indices: List[int]) -> 'Dataset':
        """
        Create a subset of the dataset.
        
        Args:
            indices: List of indices to include
            
        Returns:
            New Dataset object with subset of data
        """
        subset_x = [self.x_data[i] for i in indices if i < len(self.x_data)]
        subset_y = [self.y_data[i] for i in indices if i < len(self.y_data)]
        
        # Create temporary file for subset
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        df = pd.DataFrame({'x': subset_x, 'y': subset_y})
        df.to_csv(temp_file.name, index=False)
        
        return Dataset(temp_file.name)
    
    def __str__(self) -> str:
        """String representation of the dataset."""
        return f"Dataset({self.filepath}): {len(self.data)} samples"
    
    def __len__(self) -> int:
        """Return number of samples (operator overloading)."""
        return len(self.data) if self.data else 0
    
    def __getitem__(self, index: int) -> Tuple[float, float]:
        """Get item by index (operator overloading)."""
        if self.data and 0 <= index < len(self.data):
            return self.data[index]
        raise IndexError("Index out of range")


def validate_data(x_data: List[float], y_data: List[float]) -> Tuple[bool, str]:
    """
    Validate input data for linear regression.
    
    Args:
        x_data: List of x values
        y_data: List of y values
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(x_data) != len(y_data):
        return False, f"Data length mismatch: x={len(x_data)}, y={len(y_data)}"
    
    if len(x_data) < 2:
        return False, "Insufficient data points (need at least 2)"
    
    # Check for constant x values (would cause division by zero in regression)
    if len(set(x_data)) == 1:
        return False, "All x values are identical"
    
    return True, "Data is valid"
