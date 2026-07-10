# Decision Tree Diagnostics — Day3

Overview
--------

This project trains and visualizes a Decision Tree classifier on `machine_data.csv`, produces evaluation metrics, and saves visual outputs under the project root.

Files of interest
- [train_tree.py](train_tree.py) — train model and produce test split
- [predict.py](predict.py) — run predictions and print confusion matrix / report
- [visualize_tree.py](visualize_tree.py) — generate `tree.png`, `temperature_hist.png`, `correlation_sns.png`
- [model.py](model.py) — helper to save/load model and inspect critical cases
- [alerts.py](alerts.py) — single-sample alerting helper
- [main.py](main.py) — orchestrator for the pipeline

Setup (Windows)
-----------------
1. Create and activate a virtual environment (if you haven't):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install required packages:

```powershell
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install pandas numpy scikit-learn matplotlib scipy seaborn streamlit
```

Running the pipeline
--------------------
- Train the model and print evaluation:

```powershell
.\.venv\Scripts\python.exe train_tree.py
```

- Generate visualizations (or run visualize standalone):

```powershell
.\.venv\Scripts\python.exe visualize_tree.py
```

- Run the orchestrator (`main.py`) to execute the steps end-to-end:

```powershell
.\.venv\Scripts\python.exe main.py
```

Interactive UI (optional)
-------------------------
You can create a small Streamlit app to explore model predictions and images. To run a Streamlit app named `app.py` (example not included by default):

```powershell
.\.venv\Scripts\python.exe -m pip install streamlit
streamlit run app.py
```

Notes & Tips
-----------
- If your confusion matrix is missing classes, use `stratify=y` in `train_test_split` or compute labels dynamically in `predict.py`.
- To silence Pandas migration warnings, convert object columns to string dtype or use `select_dtypes(include=['string'])`.
- For single-sample prediction warnings, pass a DataFrame with the same feature columns to `model.predict()`.
- Images are saved in the project root: `tree.png`, `temperature_hist.png`, `correlation_sns.png`.

If you want, I can add a ready-to-run `app.py` Streamlit UI and a `requirements.txt`. Which would you prefer next?
