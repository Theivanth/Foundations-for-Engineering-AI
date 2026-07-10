# Module 2 - Day 2: Logistic Regression (Pollution Classification)

This folder contains a logistic regression example for classifying air quality into categories based on weather features.

## Files

- `main.py` - End-to-end workflow: trains the model, evaluates it, makes a sample prediction, triggers alerts, and generates plots.
- `train.py` - Loads `pollution_data.csv`, creates AQI categories, trains a `LogisticRegression` model, and returns the model, encoder, test split, and dataframe.
- `evaluate.py` - Evaluates the trained model and prints accuracy, classification report, and confusion matrix.
- `predict.py` - Helper to predict AQI category for new inputs.
- `plots.py` - Generates and saves `outputs/aqi_distribution.png` and `outputs/pollution_categories.png`.
- `alerts.py` - Simple alerting and model save/load helpers.
- `pollution_data.csv` - Sample dataset used for training.
- `model.pkl` - Saved model artifact (created at runtime).

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib

Install with:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Usage

Run the main workflow from the module directory:

```bash
cd Module2/day2_logistic_regression
python main.py
```

To run without opening plot windows (CI or headless servers):

```bash
# PowerShell
$env:MPLBACKEND='Agg'; python main.py

# Bash
MPLBACKEND=Agg python main.py
```

Outputs:

- `outputs/aqi_distribution.png`
- `outputs/pollution_categories.png`
- `model.pkl` (saved model)
- Console logs with training/evaluation metrics and alerts

## Notes

- Data file loading is resolved relative to the module so scripts run correctly from other working directories.
- The model is a simple baseline; evaluation may show low performance — consider feature engineering and hyperparameter tuning for improvements.
