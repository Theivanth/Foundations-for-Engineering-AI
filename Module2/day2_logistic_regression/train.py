import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def train_model(csv_file="pollution_data.csv"):
    # Resolve csv path relative to this module so the function works
    # regardless of current working directory.
    csv_path = Path(csv_file)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).resolve().parent / csv_file

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Create AQI category
    def classify_aqi(aqi):
        if aqi <= 150:
            return "Safe"
        elif aqi <= 250:
            return "Moderate"
        else: 
            return "Dangerous"
    df["Category"] = df["AQI"].apply(classify_aqi)

    # Encode labels
    encoder = LabelEncoder()
    df["Category"] = encoder.fit_transform(df["Category"])

    # Features and target
    X = df[["Temperature", "Humidity", "WindSpeed"]]
    y = df["Category"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training set size:", X_train.shape[0])
    print("Testing set size:", X_test.shape[0])

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("Model training completed.")

    return model, encoder, X_test, y_test, df
