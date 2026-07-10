import pandas as pd
from pathlib import Path

# Read and clean the dataset
DATA_FILE = Path(__file__).resolve().parent / "environment_data.csv"
NUMERIC_COLUMNS = ["Temperature_C", "Humidity_Percent", "AQI"]

def read_data(file_path=DATA_FILE):
    df = pd.read_csv(file_path)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].fillna(df[NUMERIC_COLUMNS].mean(numeric_only=True))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

# Standardization
def standardize(df, columns):
    for col in columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

# Global dataset used by the application
data = read_data()
