# Module 2 - Day 1: Classification & Alerting

This folder contains the Day 1 classification exercise for air quality monitoring. The scripts demonstrate loading environmental data, classifying AQI into categories, plotting status distribution, generating alerts for dangerous AQI values, and producing a simple report.

## Files

- `main.py` - Orchestrates the workflow: loads data, adds AQI status labels, shows distribution, generates alerts and writes `report.txt`.
- `label_generator.py` - Loads the dataset and provides functions to classify AQI and filter dangerous days.
- `visualize.py` - Creates a bar chart showing the AQI status distribution (`Safe`, `Moderate`, `Dangerous`).
- `alerts.py` - Prints alerts for dangerous AQI records and generates a summary `report.txt`.
- `environment.csv` - Sample dataset with daily pollutant readings and AQI values.
- `report.txt` - Generated summary report created by `main.py`.

## Requirements

- Python 3.x
- pandas
- matplotlib

Install dependencies with:

```bash
pip install pandas matplotlib
```

## Usage

Run the main script from the module directory:

```bash
cd Module2/day1_classification
python main.py
```

If you are running on a headless server (no GUI), set a non-interactive Matplotlib backend to avoid blocking plot windows:

```bash
# PowerShell
$env:MPLBACKEND='Agg'; python main.py

# Bash
MPLBACKEND=Agg python main.py
```

## Notes

- `main.py` expects `environment.csv` to be located in the same directory. `label_generator.load_data()` resolves the file path relative to the module file to make execution robust across working directories.
- Plots are shown with `plt.show()` by default; if you prefer saving images instead, update `visualize.py` to call `plt.savefig()` before `plt.close()`.
