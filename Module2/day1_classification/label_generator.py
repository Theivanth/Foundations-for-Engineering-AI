import pandas as pd
from pathlib import Path


def load_data(filename):
    """
    Load the dataset and clean column names. Resolve relative paths
    relative to this module so the function works regardless of
    the current working directory.
    """
    file_path = Path(filename)
    if not file_path.is_absolute():
        file_path = Path(__file__).resolve().parent / filename

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df


def classify_aqi(aqi):
    """
    Classify a single AQI value.
    """
    if aqi <= 100:
        return "Safe"
    elif aqi <= 150:
        return "Moderate"
    else:
        return "Dangerous"


def add_status_column(df):
    """
    Add a Status column based on AQI.
    """
    df["Status"] = df["AQI"].apply(classify_aqi)
    return df


def get_dangerous_days(df):
    """
    Return only dangerous AQI records.
    """
    return df[df["Status"] == "Dangerous"]