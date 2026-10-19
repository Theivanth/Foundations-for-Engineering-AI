"""Generate a synthetic sensor dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sensor_data(n: int = 30000, seed: int = 42) -> pd.DataFrame:
    """Create synthetic sensor readings and their healthy/warning/failure labels."""
    rng = np.random.default_rng(seed)
    temperature = rng.normal(70, 12, n)
    vibration = rng.normal(4, 2, n)
    pressure = rng.normal(110, 15, n)
    rpm = rng.normal(1500, 180, n)
    current = rng.normal(10, 3, n)
    noise = rng.normal(60, 10, n)

    risk_score = (
        0.04 * (temperature - 70) + 0.35 * (vibration - 4)
        + 0.015 * (pressure - 110) + 0.002 * (rpm - 1500)
        + 0.15 * (current - 10) + 0.04 * (noise - 60)
        + rng.normal(0, 0.6, n)
    )
    status = np.where(risk_score < 0.5, "healthy",
                      np.where(risk_score < 2.2, "warning", "failure"))
    data = pd.DataFrame({
        "temperature": temperature, "vibration": vibration, "pressure": pressure,
        "rpm": rpm, "current": current, "noise": noise, "status": status,
    })

    # Add up to 30 missing readings, spread across three sensor columns.
    missing_count = min(30, n)
    missing_rows = rng.choice(data.index, size=missing_count, replace=False)
    split1, split2 = missing_count // 3, 2 * missing_count // 3
    data.loc[missing_rows[:split1], "temperature"] = np.nan
    data.loc[missing_rows[split1:split2], "vibration"] = np.nan
    data.loc[missing_rows[split2:], "current"] = np.nan
    return data


def save_sensor_data(csv_path: str | Path = "sensor_dataset.csv", n: int = 30000,
                     seed: int = 42) -> pd.DataFrame:
    """Generate data, save it to CSV, and return the DataFrame."""
    data = generate_sensor_data(n=n, seed=seed)
    data.to_csv(csv_path, index=False)
    return data


def main() -> None:
    data = save_sensor_data()
    print(data.head())
    print(data.isnull().sum())
    print(data["status"].value_counts())


if __name__ == "__main__":
    main()
