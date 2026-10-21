from pathlib import Path
from collections import deque
from datetime import datetime
import csv
import json
import time

import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from autoencoder import SensorAutoencoder


# ---------------------------------------
# 1. CONFIGURATION
# ---------------------------------------

SEED = 42
UPDATE_SECONDS = 1.0
HISTORY_SIZE = 120

MODEL_PATH = Path(
    "saved_models/best_autoencoder.pt"
)

SCALER_PATH = Path(
    "processed_data/scaler.pkl"
)

FEATURE_PATH = Path(
    "processed_data/sensor_columns.pkl"
)

THRESHOLD_PATH = Path(
    "saved_models/evaluation/threshold.json"
)

LOG_DIR = Path("monitoring_logs")
LOG_PATH = LOG_DIR / "sensor_monitor.csv"

# ---------------------------------------
# 3. SENSOR SIMULATOR
# ---------------------------------------

class SensorSimulator:

    def __init__(self):
        self.step = 0
        self.fault = "normal"
        self.fault_step = 0
        self.rng = np.random.default_rng(SEED)

    def inject_fault(self, fault_type):
        self.fault = fault_type
        self.fault_step = 0
        print("Injected fault:", fault_type)

    def next_reading(self):

        t = self.step

        daily_cycle = np.sin(
            2 * np.pi * t / 1440
        )

        load = (
            0.60
            + 0.14 * np.sin(2 * np.pi * t / 360)
            + 0.06 * daily_cycle
            + self.rng.normal(0, 0.025)
        )

        load = np.clip(load, 0.30, 0.95)

        temperature = (
            49 + 22 * load
            + 1.3 * daily_cycle
            + self.rng.normal(0, 0.65)
        )

        vibration = (
            0.10 + 0.34 * load
            + self.rng.normal(0, 0.018)
        )

        pressure = (
            4.0 + 2.1 * load
            + self.rng.normal(0, 0.075)
        )

        motor_current = (
            5.0 + 8.2 * load
            + self.rng.normal(0, 0.20)
        )

        true_fault = self.fault

        # Gradually intensify the injected fault
        severity = min(
            1.0,
            0.2 + self.fault_step / 40
        )

        if self.fault == "overheating":
            temperature += 12 * severity

        elif self.fault == "bearing_fault":
            vibration += 0.45 * severity

        elif self.fault == "pressure_drop":
            pressure -= 1.8 * severity

        elif self.fault == "motor_overload":
            motor_current += 5 * severity
            temperature += 5 * severity

        elif self.fault == "combined_fault":
            temperature += 10 * severity
            vibration += 0.35 * severity
            motor_current += 4 * severity

        if self.fault != "normal":
            self.fault_step += 1

        self.step += 1

        values = np.array(
            [
                temperature,
                vibration,
                pressure,
                motor_current
            ],
            dtype=np.float32
        )

        return values, true_fault


# ---------------------------------------
# 4. LOAD SAVED ARTIFACTS
# ---------------------------------------

def load_artifacts():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        FEATURE_PATH,
        THRESHOLD_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model = SensorAutoencoder(
        input_dim=checkpoint["input_dim"],
        latent_dim=checkpoint["latent_dim"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURE_PATH)

    with open(
        THRESHOLD_PATH,
        encoding="utf-8"
    ) as f:
        threshold_config = json.load(f)

    threshold = float(threshold_config["threshold"])

    if not np.isfinite(threshold):
        raise ValueError("Invalid saved threshold")

    expected_features = [
        "temperature",
        "vibration",
        "pressure",
        "motor_current"
    ]

    if features != expected_features:
        raise ValueError(
            "Unexpected sensor feature order"
        )

    print("Model loaded")
    print("Threshold:", threshold)

    return model, scaler, features, threshold, device


# ---------------------------------------
# 5. MODEL INFERENCE
# ---------------------------------------

def detect_anomaly(
    values,
    model,
    scaler,
    threshold,
    device
):
    scaled = scaler.transform(
        values.reshape(1, -1)
    )

    tensor = torch.tensor(
        scaled,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        reconstructed = model(tensor)

        error = torch.mean(
            (tensor - reconstructed) ** 2,
            dim=1
        ).item()

    is_anomaly = error > threshold

    return error, is_anomaly


# ---------------------------------------
# 6. INITIALIZE MONITORING DATA
# ---------------------------------------

def initialize_log(features):
    LOG_DIR.mkdir(exist_ok=True)

    with open(
        LOG_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "sample",
            *features,
            "reconstruction_error",
            "threshold",
            "predicted_anomaly",
            "injected_fault"
        ])


# ---------------------------------------
# 7. CREATE MONITORING DASHBOARD
# ---------------------------------------

def create_dashboard(simulator, threshold):
    plt.ion()

    fig, axes = plt.subplots(
        5, 1,
        figsize=(13, 10),
        sharex=True
    )

    plt.subplots_adjust(
        bottom=0.17,
        hspace=0.45
    )

    fig.suptitle(
        "Smart Equipment Anomaly Detector",
        fontsize=15,
        fontweight="bold"
    )

    sensor_labels = [
        "Temperature (°C)",
        "Vibration (mm/s)",
        "Pressure (bar)",
        "Motor Current (A)"
    ]

    lines = []

    for i in range(4):
        line, = axes[i].plot(
            [],
            [],
            linewidth=1.5
        )

        lines.append(line)
        axes[i].set_ylabel(sensor_labels[i])
        axes[i].grid(alpha=0.3)

    error_line, = axes[4].plot(
        [],
        [],
        label="Reconstruction Error"
    )

    axes[4].axhline(
        threshold,
        color="red",
        linestyle="--",
        label="Anomaly Threshold"
    )

    axes[4].set_ylabel("MSE")
    axes[4].set_xlabel("Simulated minute")
    axes[4].legend()
    axes[4].grid(alpha=0.3)

    status_text = fig.text(
        0.02,
        0.965,
        "Status: Starting...",
        fontsize=11
    )

    fault_options = [
        ("Normal", "normal"),
        ("Heat", "overheating"),
        ("Bearing", "bearing_fault"),
        ("Pressure", "pressure_drop"),
        ("Overload", "motor_overload"),
        ("Combined", "combined_fault")
    ]

    buttons = []

    for i, (label, fault_type) in enumerate(
        fault_options
    ):
        button_ax = fig.add_axes([
            0.06 + i * 0.15,
            0.055,
            0.13,
            0.045
        ])

        button = Button(
            button_ax,
            label
        )

        def callback(event, fault=fault_type):
            simulator.inject_fault(fault)

        button.on_clicked(callback)
        buttons.append(button)

    return fig, axes, lines, error_line, status_text, buttons


# ---------------------------------------
# 8. RUN REAL-TIME MONITORING
# ---------------------------------------

def log_reading(
    values,
    error,
    is_anomaly,
    true_fault,
    sample_count,
    threshold
):
    with open(
        LOG_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            sample_count,
            *values.tolist(),
            error,
            threshold,
            int(is_anomaly),
            true_fault
        ])


def update_dashboard(
    values,
    error,
    is_anomaly,
    true_fault,
    sample_count,
    alert_count,
    times,
    sensor_history,
    error_history,
    lines,
    axes,
    error_line,
    status_text,
    threshold,
    fig
):
    times.append(sample_count)

    for i in range(4):
        sensor_history[i].append(values[i])
        lines[i].set_data(
            list(times),
            list(sensor_history[i])
        )
        axes[i].relim()
        axes[i].autoscale_view()

    error_history.append(error)
    error_line.set_data(
        list(times),
        list(error_history)
    )
    axes[4].relim()
    axes[4].autoscale_view()

    ymin, ymax = axes[4].get_ylim()
    axes[4].set_ylim(
        min(0, ymin),
        max(ymax, threshold * 1.1)
    )

    if is_anomaly:
        status = "ANOMALY DETECTED"
        status_color = "red"
    else:
        status = "NORMAL"
        status_color = "green"

    status_text.set_text(
        f"Status: {status}  |  "
        f"Error: {error:.5f}  |  "
        f"Threshold: {threshold:.5f}  |  "
        f"Alerts: {alert_count}  |  "
        f"Injected: {true_fault}"
    )
    status_text.set_color(status_color)
    fig.canvas.draw_idle()


def run_monitoring(
    simulator,
    model,
    scaler,
    threshold,
    device,
    dashboard
):
    (
        fig,
        axes,
        lines,
        error_line,
        status_text,
        buttons
    ) = dashboard

    times = deque(maxlen=HISTORY_SIZE)
    sensor_history = [
        deque(maxlen=HISTORY_SIZE)
        for _ in range(4)
    ]
    error_history = deque(maxlen=HISTORY_SIZE)
    sample_count = 0
    alert_count = 0

    print("\nMonitoring started...")
    print("Close the graph window to stop.")

    try:
        while plt.fignum_exists(fig.number):
            start_time = time.monotonic()

            values, true_fault = simulator.next_reading()
            error, is_anomaly = detect_anomaly(
                values,
                model,
                scaler,
                threshold,
                device
            )

            sample_count += 1
            if is_anomaly:
                alert_count += 1

            log_reading(
                values,
                error,
                is_anomaly,
                true_fault,
                sample_count,
                threshold
            )
            update_dashboard(
                values,
                error,
                is_anomaly,
                true_fault,
                sample_count,
                alert_count,
                times,
                sensor_history,
                error_history,
                lines,
                axes,
                error_line,
                status_text,
                threshold,
                fig
            )

            plt.pause(0.05)

            remaining = (
                UPDATE_SECONDS
                - (time.monotonic() - start_time)
            )

            if remaining > 0:
                plt.pause(remaining)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

    finally:
        plt.ioff()
        plt.close("all")
        print("Total readings:", sample_count)
        print("Total alerts:", alert_count)
        print("Log saved to:", LOG_PATH.resolve())


def main():
    model, scaler, features, threshold, device = load_artifacts()
    initialize_log(features)

    simulator = SensorSimulator()
    dashboard = create_dashboard(simulator, threshold)
    run_monitoring(
        simulator,
        model,
        scaler,
        threshold,
        device,
        dashboard
    )


if __name__ == "__main__":
    main()