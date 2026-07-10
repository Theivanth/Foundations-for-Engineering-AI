# Day 4 Preprocessing

This project prepares raw environmental data for machine learning by cleaning missing values, scaling numeric features, and encoding categorical labels.

## Files

- `raw_environment_data.csv` - raw input dataset
- `preprocessing.py` - preprocessing pipeline script
- `cleaned_data.csv` - cleaned output dataset
- `feature_engineering.py` - placeholder for future feature engineering logic
- `.venv/` - Python virtual environment

## Setup

1. Create or activate the virtual environment:
   - Windows PowerShell:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```

2. Install dependencies:
   ```powershell
   pip install scikit-learn pandas
   ```

## Usage

Run the preprocessing pipeline with input and output paths:

```powershell
python preprocessing.py --input raw_environment_data.csv --output cleaned_data.csv
```

The script prints step-by-step command-line status messages as it runs.

## Notes

- Numeric columns are scaled using `MinMaxScaler`.
- The `Humidity` column is filled with its mean if any values are missing.
- Categorical columns `Location` and `Weather` are label encoded.
