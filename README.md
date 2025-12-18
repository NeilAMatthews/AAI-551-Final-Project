# AAI-551 Final Project — Hourly Energy Consumption Forecasting (AEP)

## Link to the original dataset
https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?select=AEP_hourly.csv

## Students
- Christopher Kaldas
- Neil Mathews

## Problem
We analyze and forecast hourly electricity demand (MW) using the AEP hourly dataset. Accurate demand forecasting helps utilities plan generation, reduce cost, and improve grid reliability.

## Dataset
Source: Kaggle “Hourly Energy Consumption” (`AEP_hourly.csv`)  
Columns used:
- `Datetime` (timestamp)
- `AEP_MW` (demand in MW)

**File location in this repo:**
- `data/AEP_hourly.csv`

## Project Structure (current repo)
- `main.ipynb` : main program (Jupyter Notebook) — runs the full workflow
- `data/AEP_hourly.csv` : dataset used by the notebook
- `src/data_processing.py` : `Dataset` class + validation + metadata/statistics
- `src/timeseries.py` : `TimeSeriesDataset` (inherits from `Dataset`) + time-series feature engineering
- `src/models.py` : forecasting approaches (baselines + feature-based model)
- `src/metrics.py` : evaluation metrics (ex: MAE/RMSE/MAPE)
- `src/__pycache__/` : auto-generated Python cache files (not required for submission)

## Requirements
- Python 3.12 or 3.13
- Libraries used:
  - numpy
  - pandas
  - matplotlib
  - pytest (for unit tests)

## How to Run
1. **Install dependencies**
   - Option A (recommended if you have `requirements.txt`):
     - `pip install -r requirements.txt`
   - Option B (manual install):
     - `pip install numpy pandas matplotlib pytest`

2. **Verify dataset path**
   - Confirm the file exists here:
     - `data/AEP_hourly.csv`

3. **Run the notebook**
   - Open and run all cells in:
     - `main.ipynb`

## How to Use
1. **Open the notebook**
   - Launch Jupyter and open: `main.ipynb`

2. **Load the dataset**
   - The notebook should load the CSV from:
     - `data/AEP_hourly.csv`
   - Example usage inside the notebook:
     - `ds = TimeSeriesDataset("data/AEP_hourly.csv")`

3. **Explore the data**
   - Run the EDA/plotting cells to visualize the time series.
   - Optional: slice a date range for focused analysis (if included in your code):
     - `ds_slice = ds.slice_range("2016-01-01", "2017-01-01")`

4. **Train models**
   - Run the model cells in the notebook in order (baselines → improved model).
   - One of the approaches includes a **while-loop** tuning step (if you kept the tuner).

5. **Evaluate results**
   - The notebook computes evaluation metrics (ex: MAE/RMSE/MAPE).
   - Compare metrics across approaches to determine the best-performing method.

## How to Test (Pytest)
If your repo includes tests (recommended for the rubric), run from the project root:
- `pytest -q`

> If you do not currently have a `tests/` folder, add one (ex: `tests/test_models.py`, `tests/test_timeseries.py`) so the graders can verify your testing requirement.

## Notes for Submission (recommended cleanup)
- Do not commit `src/__pycache__/` files (add them to `.gitignore`):
  - `__pycache__/`
  - `*.pyc`

## Team Contributions
- Christopher Kaldas: <implemented src processing files>
- Neil Mathews: <implemented MAIN and also implemented src files. Researched real life implementations that we are applying this project for.>
