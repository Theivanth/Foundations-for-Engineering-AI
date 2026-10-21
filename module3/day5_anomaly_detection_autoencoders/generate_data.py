from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED = 42
N_SAMPLES = 20_000
INTERVAL = "1min"
OUTPUT_DIR = Path("data")

ANOMALY_EVENTS = [
    (12_200, 80, "overheating"),
    (13_500, 65, "bearing_fault"),
    (14_800, 90, "pressure_drop"),
    (16_200, 75, "motor_overload"),
    (17_800, 100, "combined_fault"),
    (19_000, 70, "bearing_fault"),
]

SENSOR_COLUMNS = [
    "temperature",
    "vibration",
    "pressure",
    "motor_current",
]


def generate_normal_readings(n_samples=N_SAMPLES, seed=SEED, interval=INTERVAL):
    """Generate timestamped synthetic sensor readings without anomalies."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=n_samples,
        freq=interval
    )
    t = np.arange(n_samples)

    daily_cycle = np.sin(2 * np.pi * t / 1440)
    load = (
        0.60
        + 0.14 * np.sin(2 * np.pi * t / 360)
        + 0.06 * daily_cycle
        + rng.normal(0, 0.025, n_samples)
    )
    load = np.clip(load, 0.30, 0.95)

    return pd.DataFrame({
        "timestamp": timestamps,
        "temperature": (
            49 + 22 * load + 1.3 * daily_cycle
            + rng.normal(0, 0.65, n_samples)
        ),
        "vibration": (
            0.10 + 0.34 * load
            + rng.normal(0, 0.018, n_samples)
        ),
        "pressure": (
            4.0 + 2.1 * load
            + rng.normal(0, 0.075, n_samples)
        ),
        "motor_current": (
            5.0 + 8.2 * load
            + rng.normal(0, 0.20, n_samples)
        ),
        "anomaly": np.zeros(n_samples, dtype=int),
        "anomaly_type": np.full(n_samples, "normal", dtype=object)
    })


def inject_anomalies(df, anomaly_events=ANOMALY_EVENTS):
    """Inject labeled synthetic faults into a copy of the readings."""
    df = df.copy()
    n_samples = len(df)

    for start, duration, fault_type in anomaly_events:
        if start < 0 or duration < 0:
            raise ValueError("Anomaly starts and durations must be non-negative")

        end = min(start + duration, n_samples)
        idx = np.arange(start, end)
        if not len(idx):
            continue

        severity = np.linspace(0.2, 1.0, len(idx))

        if fault_type == "overheating":
            df.loc[idx, "temperature"] += 12 * severity
        elif fault_type == "bearing_fault":
            df.loc[idx, "vibration"] += 0.45 * severity
        elif fault_type == "pressure_drop":
            df.loc[idx, "pressure"] -= 1.8 * severity
        elif fault_type == "motor_overload":
            df.loc[idx, "motor_current"] += 5.0 * severity
            df.loc[idx, "temperature"] += 5.0 * severity
        elif fault_type == "combined_fault":
            df.loc[idx, "temperature"] += 10.0 * severity
            df.loc[idx, "vibration"] += 0.35 * severity
            df.loc[idx, "motor_current"] += 4.0 * severity
        else:
            raise ValueError(f"Unsupported anomaly type: {fault_type}")

        df.loc[idx, "anomaly"] = 1
        df.loc[idx, "anomaly_type"] = fault_type

    return df


def validate_dataset(df, n_samples=N_SAMPLES):
    """Validate generated rows, labels, and chronological timestamps."""
    if len(df) != n_samples:
        raise ValueError(
            f"Expected {n_samples} samples, got {len(df)}"
        )
    if df.isnull().any().any():
        raise ValueError("Generated dataset contains missing values")

    anomaly_start = int(n_samples * 0.60)
    if not (df.iloc[:anomaly_start]["anomaly"] == 0).all():
        raise ValueError("Anomalies must not occur in the first 60% of data")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps must be monotonically increasing")


def save_dataset(df, output_dir=OUTPUT_DIR):
    """Save the generated dataset as a CSV and return its path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "industrial_sensor_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def print_dataset_summary(df, csv_path):
    """Print dataset counts, example rows, and sensor statistics."""
    print("\nDataset generated successfully!")
    print("Saved to:", csv_path.resolve())
    print("\nDataset shape:")
    print(df.shape)
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))
    print("\nAnomaly distribution:")
    print(df["anomaly"].value_counts())
    print("\nAnomaly types:")
    print(df["anomaly_type"].value_counts())
    print("\nSensor statistics:")
    print(df[SENSOR_COLUMNS].describe())


def plot_dataset(df, output_dir=OUTPUT_DIR, show=True):
    """Plot sensor data and injected anomalies; return the saved plot path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    anomalies = df[df["anomaly"] == 1]

    fig, axes = plt.subplots(
        4, 1,
        figsize=(14, 10),
        sharex=True
    )

    for ax, sensor in zip(axes, SENSOR_COLUMNS):
        ax.plot(
            df["timestamp"],
            df[sensor],
            linewidth=0.7,
            label=sensor
        )
        ax.scatter(
            anomalies["timestamp"],
            anomalies[sensor],
            color="red",
            s=8,
            label="Injected anomaly"
        )
        ax.set_ylabel(sensor)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Timestamp")
    plt.suptitle("Synthetic Industrial Equipment Sensor Data")
    plt.tight_layout()

    plot_path = output_dir / "sensor_data_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return plot_path


def generate_dataset(
    n_samples=N_SAMPLES,
    seed=SEED,
    interval=INTERVAL,
    anomaly_events=ANOMALY_EVENTS,
    output_dir=OUTPUT_DIR,
    plot=True,
    show=True
):
    """Generate, validate, and save the synthetic dataset and optional plot."""
    df = generate_normal_readings(n_samples, seed, interval)
    df = inject_anomalies(df, anomaly_events)
    validate_dataset(df, n_samples)

    csv_path = save_dataset(df, output_dir)
    print_dataset_summary(df, csv_path)

    plot_path = None
    if plot:
        plot_path = plot_dataset(df, output_dir, show)
        print("\nPlot saved to:", plot_path.resolve())

    return {
        "data": df,
        "csv_path": csv_path,
        "plot_path": plot_path
    }


def main():
    """Generate the default dataset and display its plot."""
    generate_dataset()


if __name__ == "__main__":
    main()
