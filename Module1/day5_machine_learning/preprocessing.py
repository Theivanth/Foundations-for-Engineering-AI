import pandas as pd
def preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Handle missing values (example: fill with mean)
    df.fillna(df.mean(), inplace=True)
    
    # Encode categorical variables (example: one-hot encoding)
    df = pd.get_dummies(df, drop_first=True)
    
    # Normalize numerical features (example: min-max scaling)
    # Do not scale the target column 'AQI' so predictions remain on the original scale
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'AQI' in numerical_cols:
        numerical_cols = [c for c in numerical_cols if c != 'AQI']

    # Record original min/max for the numerical features so we can scale
    # new/future inputs consistently when making predictions.
    scalers = {}
    if numerical_cols:
        orig_mins = df[numerical_cols].min()
        orig_maxs = df[numerical_cols].max()
        for col in numerical_cols:
            scalers[col] = {'min': float(orig_mins[col]), 'max': float(orig_maxs[col])}

        df[numerical_cols] = (df[numerical_cols] - orig_mins) / (orig_maxs - orig_mins)

    return df, scalers

