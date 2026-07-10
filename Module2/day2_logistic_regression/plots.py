import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Load data relative to this module
DATA_FILE = Path(__file__).resolve().parent / "pollution_data.csv"
df = pd.read_csv(DATA_FILE)


def classify_aqi(aqi):
    # Simple categorization (adjust thresholds as needed)
    if aqi <= 150:
        return "Safe"
    elif aqi <= 250:
        return "Moderate"
    else:
        return "Dangerous"


df["Category"] = df["AQI"].apply(classify_aqi)


def plot_aqi_trends(df, out_dir: Path = Path("outputs")):
    """Plot AQI distribution and category counts, save figures to `out_dir`."""
    out_dir.mkdir(exist_ok=True)

    # Category counts
    print("Category Counts:")
    print(df["Category"].value_counts())

    # AQI Distribution Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df["AQI"], bins=10, edgecolor="black")
    plt.title("AQI Distribution")
    plt.xlabel("AQI")
    plt.ylabel("Frequency")
    hist_path = out_dir / "aqi_distribution.png"
    plt.savefig(hist_path)
    plt.close()
    print(f"Saved AQI distribution to: {hist_path}")

    # Pollution Categories Bar Chart
    counts = df["Category"].value_counts()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values)
    plt.title("Pollution Categories")
    plt.xlabel("Category")
    plt.ylabel("Count")
    bar_path = out_dir / "pollution_categories.png"
    plt.savefig(bar_path)
    plt.close()
    print(f"Saved pollution categories chart to: {bar_path}")


if __name__ == "__main__":
    plot_aqi_trends(df)

