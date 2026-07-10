# Environmental Dashboard

This project analyzes environmental data from `environment_data.csv` and provides both script-based reports and a Streamlit GUI dashboard.

## Requirements

- Python 3.14
- A virtual environment in the workspace

## Setup

1. Activate the workspace venv:

```powershell
cd "c:\Users\Theivanth\AI_ML_Environmental\day3_visualization"
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run the script

Use `dashboard.py` to print summary statistics, detect outliers, and generate plots saved under `outputs/`.

```powershell
python dashboard.py --threshold 1.5
```

- `--threshold` sets the outlier cutoff as a multiple of the temperature standard deviation.
- Example: `--threshold 1.5` finds points where temperature exceeds average by 1.5× standard deviation.

## Run the Streamlit GUI

Start the dashboard:

```powershell
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

### Change the threshold in the GUI

- Use the sidebar slider labelled `Outlier threshold (× std)`.
- The app automatically updates the outlier table and plots.

## Project files

- `dashboard.py` — script-based analysis and plot generation
- `app.py` — Streamlit GUI dashboard
- `correlation.py` — correlation matrix computation and plotting
- `plots.py` — helper plotting functions
- `environment_data.csv` — sample environmental dataset
- `requirements.txt` — Python package dependencies

## Notes

- `outputs/` is used to save generated images from `dashboard.py`.
- If you modify the GUI or script, restart Streamlit or rerun the script as needed.
