# Day 1: Neural Network Fundamentals

This folder explores sensor-based equipment health classification. It includes
a synthetic sensor dataset, data cleaning and splitting helpers, a logistic
regression baseline, and a small PyTorch neural network.

## Requirements

- Python 3.10 or newer
- Packages listed in [`requirements.txt`](requirements.txt)

## Set up

Use the existing `m1` environment; this project does not need a separate
virtual environment. Open PowerShell in this folder and install dependencies
into `m1`:

```powershell
$PYTHON = "$HOME\.venvs\m1\Scripts\python.exe"
& $PYTHON -m pip install -r requirements.txt
```

The included `sensor_dataset.csv` is used by default. To generate a fresh
30,000-row synthetic dataset:

```powershell
& $PYTHON generate.py
```

## Run the examples

Inspect and clean the sensor data:

```powershell
& $PYTHON inspect_data.py
```

Run the logistic-regression baseline and produce train/validation/test splits:

```powershell
& $PYTHON model.py
& $PYTHON train.py
```

Train the PyTorch neural network, print its test accuracy, and enter six sensor
readings for an interactive condition prediction:

```powershell
& $PYTHON neural.py
```

The interactive predictor prompts for temperature (Celsius), vibration (mm/s),
pressure (kPa), RPM, current (A), and noise (dB). Its class probabilities are
model estimates, not a substitute for equipment inspection or safety controls.

## Project files

- `generate.py` creates the synthetic dataset.
- `inspect_data.py` fills missing values, clips outliers, and filters invalid
  sensor readings.
- `model.py` provides data preparation and a logistic-regression baseline.
- `train.py` demonstrates train/validation/test splitting.
- `create_model.py` defines the PyTorch feed-forward network.
- `pytorch_data.py` creates PyTorch datasets and data loaders.
- `neural.py` trains the neural network and runs an interactive prediction.
