# KNN Water Quality Classification

This module trains and evaluates a K-Nearest Neighbors model to classify water quality as `Clean`, `Polluted`, or `Hazardous` using environmental sensor measurements.

## Project Structure

- `main.py` - Entry point that loads data, trains the model, evaluates performance, tunes `k`, and displays a sample prediction.
- `train_knn.py` - Loads the dataset, splits it into training/test sets, and trains the KNN model using a scikit-learn pipeline.
- `evaluate.py` - Evaluates the trained model on test data and tunes the value of `k` using cross-validation.
- `predict.py` - Builds a sample input, gets a prediction, and returns the nearest training samples.
- `alerts.py` - Prints a user-facing message based on the predicted water quality.
- `water_quality.csv` - Dataset used for training and evaluating the model.
- `outputs/` - Saved evaluation artifacts such as the K selection plot.

## Installation

Use a Python 3.10+ environment and install dependencies if not already available.

```powershell
python -m pip install pandas scikit-learn matplotlib
```

## Run the Project

From the `Module2/day5_knn_water_quality` directory:

```powershell
python main.py
```

## Outputs

- `outputs/k_selection.png` — plot of cross-validated F1 score for selected `k` values.
- Console output showing dataset summary, classification report, accuracy, confusion matrix, predicted quality, and nearest neighbor samples.

## Notes

- The code uses the dataset path relative to the module directory, so run the script from within `Module2/day5_knn_water_quality` or use a Python environment that resolves relative imports correctly.
- The KNN tuning logic adapts `k` to the size of the training set to avoid invalid neighbor values.
