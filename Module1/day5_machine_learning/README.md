# Module 1 - Day 5: Machine Learning

This folder contains the Day 5 machine learning exercise for the environmental data project. It demonstrates preprocessing, model training, visualization, prediction, and alerting for environmental air quality data.

## Files

- `main.py` - Main workflow that preprocesses data, plots trends, trains a regression model, makes a future AQI prediction, and displays an alert.
- `preprocessing.py` - Loads data, fills missing values, encodes categorical variables, and scales numeric features.
- `prediction.py` - Trains a linear regression model on temperature and humidity to predict AQI and reports mean absolute error.
- `visualization.py` - Creates and saves plots for temperature trend, AQI trend, and the correlation matrix.
- `alert.py` - Prints an AQI-based alert message based on the latest dataset value.
- `environment_data.csv` - Sample environmental dataset used for training and analysis.
- `outputs/` - Folder for saving generated plot images.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib

Install dependencies with:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Usage

Run the main script from this folder:

```bash
python main.py
```

The script will:

- Load and preprocess the sample data
- Generate temperature and AQI trend plots
- Output a correlation matrix
- Train a linear regression model to predict AQI
- Print model evaluation metrics
- Generate a prediction for future input values
- Print an AQI alert message

## Notes

- The sample input for future predictions is currently hardcoded in `main.py`.
- Plots are saved to the `outputs/` directory.
- The model uses `Temperature` and `Humidity` as features to predict `AQI`.
