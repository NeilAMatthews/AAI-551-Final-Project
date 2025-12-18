# AAI-551-Final-Project

## Link to the original data set
https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?select=AEP_hourly.csv

# Hourly Energy Consumption Forecasting (AEP)

## Students
- Christopher Kaldas
- Neil Mathews

## Problem
We analyze and forecast hourly electricity demand (MW) using the AEP hourly dataset. Accurate demand forecasting helps utilities plan generation, reduce cost, and improve grid reliability.

## Dataset
Source: Kaggle “Hourly Energy Consumption” (AEP_hourly.csv).  
Columns used:
- `Datetime` (timestamp)
- `AEP_MW` (demand in MW)

## Project Structure
- `notebooks/main.ipynb` : main program (Jupyter Notebook) + explanation of project usages
- `data/AEP_hourly` : csv file all information came from
- `src/data_processing.py` : `Dataset` class + validation + metadata/statistics
- `src/timeseries.py` : `TimeSeriesDataset` (inherits from `Dataset`) + time-series feature engineering
- `src/models.py` : baseline models (naive, moving average + tuner) + linear regression forecaster
- `src/metrics.py` : MAE/RMSE/MAPE evaluation metrics
- `src/visualization.py` : matplotlib plotting functions



## Requirements
- Python 3.12 or 3.13
- Dependencies:
  - pandas, numpy, matplotlib, pytest

## How to Run
1. Install dependencies:
   - `pip install -r requirements.txt`

2. Ensure the dataset file is present:
   - Place `AEP_hourly.csv` in the project root (same level as `src/`).

3. Open the notebook and run all cells:
   - `notebooks/main.ipynb`

## How to Use
1. **Open the notebook**
   - Launch Jupyter and open: `notebooks/main.ipynb`

2. **Load the dataset**
   - Make sure `AEP_hourly.csv` is in the project root.
   - In the notebook, run the first cells to create the dataset object:
     - `ds = TimeSeriesDataset("AEP_hourly.csv")`
   - This parses the `Datetime` column, removes invalid/missing rows (if any), and computes summary metadata.

3. **Explore the data**
   - Run the plotting cells to visualize the time series and confirm data looks correct.
   - Optional: slice a date range for focused analysis:
     - `ds_slice = ds.slice_range("2016-01-01", "2017-01-01")`

4. **Train models**
   - Run the model cells in order:
     - Naive baseline: `NaiveLastValueForecaster`
     - Tuned moving average baseline: `tune_moving_average_window` (uses a **while loop** to choose the best window)
     - Feature-based model: `LinearRegressionForecaster` using calendar + lag features

5. **Evaluate results**
   - Metrics computed:
     - MAE, RMSE, MAPE
   - Compare the `results` dictionary to see which approach performs best.

6. **Generate outputs**
   - The notebook writes results to `outputs/`:
     - `outputs/metrics.json` (model metrics)
     - `outputs/predictions_baselines.csv` (naive + moving average predictions)
     - `outputs/predictions_linreg.csv` (linear regression predictions)

7. **Run tests (recommended before final submission)**
   - From the project root:
     - `pytest -q`
   - This verifies dataset loading/feature creation and model behavior.

## How to Test
- Run:
  - `pytest -q`

## Team Contributions
- <Name 1>: <what you did (modules, modeling, notebook, testing, etc.)>
- <Name 2>: <what you did (modules, modeling, notebook, testing, etc.)>
