import os
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from correlation import correlation_analysis
from plots import plot_temperature, plot_aqi, plot_humidity

# Correct filename for the provided dataset
DATA_FILE = "environment_data.csv"

parser = argparse.ArgumentParser(description="Run environmental data dashboard script")
parser.add_argument("--threshold", type=float, default=1.0, help="Outlier threshold in multiples of std deviation")
args = parser.parse_args()
threshold = args.threshold

df = pd.read_csv(DATA_FILE)
print(f"Using outlier threshold = {threshold} * std")
print("Environmental Data Summary:\n")
print(df.describe())

# Temperature Analysis
threshold = 1
average_temperature = df["Temperature"].mean()
max_temperature = df["Temperature"].max()
min_temperature = df["Temperature"].min()
std_temperature = df["Temperature"].std()
print(f"\nAverage Temperature: {average_temperature:.2f} °C")
print(f"Max Temperature: {max_temperature:.2f} °C")
print(f"Min Temperature: {min_temperature:.2f} °C")

# AQI Analysis
average_aqi = df["AQI"].mean()
max_aqi = df["AQI"].max()
min_aqi = df["AQI"].min()
std_aqi = df["AQI"].std()
print(f"\nAverage AQI: {average_aqi:.2f}")
print(f"Max AQI: {max_aqi:.2f}")
print(f"Min AQI: {min_aqi:.2f}")

# Humidity Analysis
average_humidity = df["Humidity"].mean()
max_humidity = df["Humidity"].max()
min_humidity = df["Humidity"].min()
std_humidity = df["Humidity"].std()
print(f"\nAverage Humidity: {average_humidity:.2f} %")
print(f"Max Humidity: {max_humidity:.2f} %")
print(f"Min Humidity: {min_humidity:.2f} %")

# Alert System
if max_aqi > 100:
    print("\nAlert: AQI levels are unhealthy! Consider taking precautions.")
if max_humidity > 80:
    print("\nAlert: Humidity levels are high! Consider taking precautions.")
if min_humidity < 30:
    print("\nAlert: Humidity levels are low! Consider taking precautions.")

# Outlier Detection
print("\nOutlier Detection:")
outlier = df[df["Temperature"] - average_temperature > threshold * std_temperature]
print(outlier)

# Ensure outputs directory exists before saving plots
os.makedirs("outputs", exist_ok=True)

# Generate plots and correlation matrix
correlation_analysis(df)
plot_temperature(df)
plot_aqi(df)
plot_humidity(df)
