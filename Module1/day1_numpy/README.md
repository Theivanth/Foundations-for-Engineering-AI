# Module 1 - Day 1: NumPy Fundamentals

This directory contains the Day 1 NumPy exercises for the environmental data project. It demonstrates basic NumPy usage for analyzing temperature data and detecting simple anomalies.

## Files

- `anomaly_detection.py` - Reads a user-supplied temperature value, appends it to a sample dataset, computes statistics, and flags the new reading as a high/low anomaly using `mean ± std`.
- `basics.py` - Computes and prints the mean, max, min, and standard deviation for a sample temperature dataset.
- `temperature_analysis.py` - Computes and prints the mean, max, min, standard deviation, and high/low thresholds for a sample temperature dataset.

## Requirements

- Python 3.x
- NumPy

Install NumPy with:

```bash
pip install numpy
```

## Usage

Run any script from this folder with Python:

```bash
python anomaly_detection.py
python basics.py
python temperature_analysis.py
```

`anomaly_detection.py` will prompt for a new temperature input.

## Notes

- These scripts are designed for learning basic NumPy array operations and simple statistical analysis.
- The anomaly detection logic in `anomaly_detection.py` uses one standard deviation from the mean as the anomaly threshold.
- The dataset values are hardcoded for demonstration purposes.
