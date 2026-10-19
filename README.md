# Foundations for Engineering AI

This repository contains a collection of environmental AI and machine learning projects organized into three modules:

- `Module1/` - foundational data science tasks using NumPy, Pandas, preprocessing, visualization, and machine learning pipelines.
- `Module2/` - classification and predictive modeling exercises using logistic regression, decision trees, random forests, and K-Nearest Neighbors.
- `module3/` - neural network fundamentals, computer vision with CNNs, and pedestrian detection.

Each subfolder contains a standalone project with its own dataset, scripts, and `requirements.txt` file.

## Repository Structure

- `Module1/day1_numpy/` - NumPy-based anomaly detection, basic calculations, and temperature analysis.
- `Module1/day2_pandas/` - Data analysis and reporting with Pandas.
- `Module1/day3_visualization/` - Visualization exercises producing charts from environmental data.
- `Module1/day4_preprocessing/` - Data cleaning and feature engineering workflows.
- `Module1/day5_machine_learning/` - Introductory machine learning pipeline with prediction and alerting.
- `Module1/final_project/` - Final integrated environmental AI project.

- `Module2/day1_classification/` - Simple classification and alert generation using environmental labels.
- `Module2/day2_logistic_regression/` - Logistic regression modeling, evaluation, and plots.
- `Module2/day3_decision_tree/` - Decision tree training, prediction, and visualization.
- `Module2/day4_random_forest/` - Random forest training, evaluation, and scoring.
- `Module2/day5_knn_water_quality/` - KNN water quality classification and tuning.

- `module3/day1_neural_network_fundamentals/` - Sensor-based equipment health classification with logistic regression and a PyTorch neural network.
- `module3/day2_computer_vision_CNNs/` - Casting defect classification with a convolutional neural network.
- `module3/day3_pedestrian_detection/` - Pedestrian detection and counting with pretrained and custom object detectors.

## Getting Started

The repository is designed to be run at the module level. Each folder has its own `requirements.txt` and should be run from inside that folder.

### Recommended Python version

- Python 3.10 or newer

### Install dependencies per module

From within a module folder, run:

```powershell
python -m pip install -r requirements.txt
```

## Running a Module

Each project typically includes a `main.py` script. Run it from the module folder like this:

```powershell
cd Module2/day5_knn_water_quality
python main.py
```

If a module includes a `README.md`, refer to it for module-specific instructions and expected outputs.

## Notes

- The repository includes datasets and output artifacts for each project.
- Generated Python cache files and virtual environments are excluded from git tracking using `.gitignore`.
- If a module uses `matplotlib`, outputs are often written to a local `outputs/` directory.

## Contribution

This repository is best explored one module at a time. Contributions can include:

- improving data loading and preprocessing
- adding better evaluation metrics and visualizations
- refactoring module code for reuse

## License

This repository does not include a formal license file. Add one if you want to publish or share the code publicly.
