# Environmental Data Analysis and Prediction System

This repository is a Python application for analyzing environmental data, displaying trends, and predicting future Air Quality Index (AQI) values.

## Project Structure

- `main.py` - Application entry point. Loads data, prints the latest AQI and alert, predicts a future AQI value, and shows visualizations.
- `preprocessing.py` - Reads and cleans `environment_data.csv`, converts numeric fields, and parses dates.
- `prediction.py` - Trains a linear regression model using temperature and humidity features, and predicts future AQI.
- `alerts.py` - Determines the latest AQI and generates an alert message.
- `visualization.py` - Plots temperature, AQI, and humidity trends over time.
- `environment_data.csv` - Sample dataset used by the application.
- `outputs/` - Optional output folder for charts or exported results.

## Dataset Schema

The CSV file uses these columns:

- `Date` - Date of the measurement
- `Temperature_C` - Temperature in degrees Celsius
- `Humidity_Percent` - Relative humidity percentage
- `AQI` - Air Quality Index value

## Requirements

- Python 3.8 or later
- `pandas`
- `scikit-learn`
- `matplotlib`

Install dependencies with:

```powershell
.venv\Scripts\python.exe -m pip install pandas scikit-learn matplotlib
```

## Run the App

Activate the virtual environment if needed, then run:

```powershell
.venv\Scripts\python.exe main.py
```

## Notes

- The dataset contains missing and invalid numeric values; preprocessing converts them to numeric and fills missing values with column means.
- The current prediction uses hard-coded sample values in `main.py`.
- To extend the app, add command-line input, improve the model, or export results to `outputs/`.
