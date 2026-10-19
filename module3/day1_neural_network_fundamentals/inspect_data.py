import pandas as pd


def clean_sensor_data(data: pd.DataFrame) -> pd.DataFrame:
    """Fill missing sensor readings, clip outliers, and remove invalid rows."""
    df = data.copy()
    features = [
        "temperature",
        "vibration",
        "pressure",
        "rpm",
        "current",
        "noise",
    ]

    # Fill missing values using the median for each sensor feature.
    for col in features:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    # Clip extreme values using the IQR method.
    for col in features:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    # Remove rows with impossible physical sensor values.
    valid_rows = (
        df["temperature"].between(-20, 200)
        & df["vibration"].between(0, 50)
        & df["pressure"].between(0, 500)
        & df["rpm"].between(0, 5000)
        & df["current"].between(0, 30)
        & df["noise"].between(0, 150)
    )
    return df.loc[valid_rows].copy()


def main() -> None:
    df = pd.read_csv("sensor_dataset.csv")
    cleaned_df = clean_sensor_data(df)
    features = [
        "temperature",
        "vibration",
        "pressure",
        "rpm",
        "current",
        "noise",
    ]

    print("Cleaned rows:", len(cleaned_df))
    print(
        "Missing values after cleaning:\n",
        cleaned_df[features + ["status"]].isnull().sum(),
    )
    print("\nSummary after cleaning:\n", cleaned_df[features].describe().round(2))
    print("\nStatus distribution:\n", cleaned_df["status"].value_counts())
    print(cleaned_df.shape)


if __name__ == "__main__":
    main()
