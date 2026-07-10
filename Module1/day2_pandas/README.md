# Module 1 - Day 2: Pandas Data Analysis

This directory contains the Day 2 Pandas exercise for the environmental data project. It demonstrates basic data loading, cleaning, analysis, and visualization using Pandas and Matplotlib.

## Files

- `analyzer.py` - Loads `environment_data.csv`, fills missing values with column means, summarizes the dataset, finds days with high AQI values, and plots AQI over time.
- `environment_data.csv` - Sample environmental dataset containing daily measurements for temperature, humidity, and air quality index (AQI).

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib

Install dependencies with:

```bash
pip install pandas numpy matplotlib
```

## Usage

Run the analysis script from this folder:

```bash
python analyzer.py
```

The script will:

- Load the sample environmental dataset
- Display descriptive statistics for all numeric columns
- Fill missing values using each column's mean
- Print rows where `AQI` is above 100
- Show a bar chart of AQI over time

## Notes

- The dataset includes a missing AQI value that is imputed with the mean AQI.
- The plot requires a graphical display environment.
- Day 2 focuses on exploring Pandas dataframes and simple visualization.
