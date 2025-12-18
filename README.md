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

## Instructions
1. Clone/open the repository on your machine.
2. Confirm the dataset file exists at:
   - `data/AEP_hourly.csv`
3. Open the Jupyter Notebook:
   - `main.ipynb`
4. Run the notebook cells from top to bottom.
5. Review the notebook outputs:
   - dataset metadata
   - validation results
   - plots/EDA
   - train/test split results
   - model evaluation metrics
6. Run the tests from the project root:
   - `pytest -q`


## Team Contributions
- Christopher Kaldas: <implemented src processing files>
- Neil Mathews: <implemented MAIN and also implemented src files. Researched real life implementations that we are applying this project for.>
