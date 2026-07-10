import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def load_data(input_path: str) -> pd.DataFrame:
    print(f"[1/5] Loading raw data from '{input_path}'")
    data = pd.read_csv(input_path)
    print(f"    Loaded {len(data)} rows and {len(data.columns)} columns")
    return data


def clean_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    print("[2/5] Cleaning missing values")
    missing_before = data.isnull().sum().sum()
    print(f"    Missing values before: {missing_before}")
    cleaned = data.copy()
    if "Humidity" in cleaned.columns:
        cleaned["Humidity"] = cleaned["Humidity"].fillna(cleaned["Humidity"].mean())
    missing_after = cleaned.isnull().sum().sum()
    print(f"    Missing values after: {missing_after}")
    return cleaned


def scale_features(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    print("[3/5] Scaling numeric features")
    scaled = data.copy()
    scaler = MinMaxScaler()
    for column in columns:
        if column in scaled.columns:
            print(f"    Scaling '{column}'")
            scaled[column] = scaler.fit_transform(scaled[[column]])
        else:
            print(f"    Warning: '{column}' not found in data")
    return scaled


def encode_labels(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    print("[4/5] Encoding categorical labels")
    encoded = data.copy()
    encoder = LabelEncoder()
    for column in columns:
        if column in encoded.columns:
            print(f"    Encoding '{column}'")
            encoded[column] = encoder.fit_transform(encoded[column].astype(str))
        else:
            print(f"    Warning: '{column}' not found in data")
    return encoded


def save_data(data: pd.DataFrame, output_path: str) -> None:
    print(f"[5/5] Saving cleaned data to '{output_path}'")
    data.to_csv(output_path, index=False)
    print("    Save complete")


def run_pipeline(input_path: str, output_path: str) -> None:
    data = load_data(input_path)
    data = clean_missing_values(data)
    data = scale_features(data, ["Humidity", "Temperature", "AQI", "CO2"])
    data = encode_labels(data, ["Location", "Weather"])
    save_data(data, output_path)
    print("Pipeline complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the preprocessing pipeline and show status updates on the command line"
    )
    parser.add_argument(
        "--input",
        default="raw_environment_data.csv",
        help="Path to raw input CSV file",
    )
    parser.add_argument(
        "--output",
        default="cleaned_data.csv",
        help="Path to output CSV file",
    )
    args = parser.parse_args()
    run_pipeline(args.input, args.output)

