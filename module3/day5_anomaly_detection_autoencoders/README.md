# Industrial Sensor Anomaly Detection with Autoencoders

A learning project that generates synthetic industrial sensor readings, preprocesses them, trains a PyTorch autoencoder, and detects anomalies using reconstruction error. It also includes an interactive monitor that simulates sensor readings and fault injection.

This is an educational simulation. The generated data is synthetic; the project does not connect to real equipment and its predictions are not operational safety guidance.

## Requirements

- Python 3.10 or newer
- Packages listed in [`requirements.txt`](requirements.txt)

Install dependencies from this directory:

```powershell
python -m pip install -r requirements.txt
```

## Run the pipeline

Run commands from this project directory so the default relative paths resolve correctly.

### 1. Generate synthetic sensor data

```powershell
python generate_data.py
```

This creates `data/industrial_sensor_data.csv` and a sensor plot at `data/sensor_data_plot.png`. The default dataset contains 20,000 timestamped records, with simulated faults in the final 40%.

### 2. Prepare the data

```powershell
python prepare_data.py
```

The script chronologically splits the data, fits `StandardScaler` using normal training records only, and saves the tensors and preprocessing artifacts in `processed_data/`.

### 3. Train the autoencoder

```powershell
python train.py
```

Training uses normal records and saves the best validation checkpoint to `saved_models/best_autoencoder.pt`, along with a loss plot.

### 4. Evaluate the model

```powershell
python evaluate.py
```

The threshold is calibrated from normal validation reconstruction errors, then evaluated against the held-out test set. Metrics, predictions, threshold configuration, and plots are saved in `saved_models/evaluation/`.

### 5. Run the interactive monitor

```powershell
python monitor.py
```

The monitor loads the saved model, scaler, feature order, and threshold. Use the buttons below the live charts to simulate normal operation or inject a fault. Readings are appended to `monitoring_logs/sensor_monitor.csv`; close the chart window or press Ctrl+C to stop.

Run these steps in order when starting from a clean checkout. Later steps require the artifacts created by earlier steps.

## Use functions from another Python file

Each script exposes callable functions and only runs its default workflow when invoked as a script. For example:

```python
from generate_data import generate_dataset
from prepare_data import prepare_data
from train import train_model
from evaluate import evaluate_model

generated = generate_dataset(plot=False)
prepared = prepare_data()
training = train_model(show_plot=False)
evaluation = evaluate_model(show_plots=False)
```

The functions accept optional input/output paths and settings; see their definitions for available parameters and return values. Run `python autoencoder.py` to print and check the example model architecture.

## Project files

- `generate_data.py` — create synthetic readings and simulated anomaly labels.
- `prepare_data.py` — validate, split, scale, and save training data.
- `autoencoder.py` — define the sensor autoencoder and provide a model demo.
- `train.py` — train the autoencoder and save the best checkpoint.
- `evaluate.py` — select a threshold, calculate test metrics, and save reports.
- `monitor.py` — visualize simulated real-time readings and anomaly alerts.

## License and citations

The project is released under the MIT License; see [`LICENSE`](LICENSE). Sources and attribution are listed in [`CITATIONS.md`](CITATIONS.md), and project citation metadata is in [`CITATION.cff`](CITATION.cff).
