# CodeAlpha Data Science Internship

<p align="center">
  <img src="assets/codealpha-data-science-banner.gif" alt="Animated blue and green CodeAlpha Data Science Internship banner" width="100%">
</p>

<p align="center">
  <strong>Four practical projects in data cleaning, exploratory analysis, classification, and regression.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-2563EB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10 or newer">
  <img src="https://img.shields.io/badge/Pandas-Analysis-16A085?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-22A06B?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Matplotlib-Visualizations-2F80ED?style=for-the-badge" alt="Matplotlib">
</p>

## ✨ Project overview

This repository contains my **CodeAlpha Data Science Internship** projects. Each task is self-contained: its Python script and supplied dataset are in that task's folder, and running the script creates the analysis outputs in that folder's `outputs/` directory.

| Project | Goal | Main result |
| --- | --- | --- |
| 🌸 **Task 1 — Iris Flower Classification** | Classify an Iris species from four flower measurements. | Logistic regression reached **93.3% test accuracy**. |
| 📈 **Task 2 — Unemployment Analysis** | Clean and explore Indian unemployment data, including the early COVID-19 period. | The state-level mean rose from **9.23%** in Jan–Feb to **22.75%** in Apr–May 2020. |
| 🚘 **Task 3 — Car Price Prediction** | Predict used-car selling prices from vehicle details. | Gradient boosting achieved **R² 0.941** on the holdout set. |
| 📣 **Task 4 — Advertising Sales Prediction** | Estimate sales from TV, radio, and newspaper advertising spend. | Random forest achieved **R² 0.979** on the holdout set. |

## 🧭 Tasks and methods

### 🌸 Task 1 — Iris Flower Classification

**Goal:** predict whether a flower is Iris-setosa, Iris-versicolor, or Iris-virginica using its sepal and petal measurements.

- Uses the four measurement columns from `Iris.csv`; the row ID is excluded from model features.
- Makes a stratified 80/20 train-test split.
- Compares logistic regression, k-nearest neighbors, and random forest using five-fold macro-F1 on the training split.
- Selects **logistic regression** and reports test accuracy, macro-F1, a class-level report, and a confusion matrix.

**Holdout result:** accuracy **0.933**, macro-F1 **0.933**.

**Files:** [`iris_flower_classification.py`](Task%201/iris_flower_classification.py) · [`metrics.json`](Task%201/outputs/metrics.json) · [`confusion_matrix.png`](Task%201/outputs/confusion_matrix.png)

### 📈 Task 2 — Unemployment Analysis with Python

**Goal:** clean and explore monthly unemployment estimates, compare rural and urban patterns, and examine the early COVID-19 period.

- Cleans inconsistent whitespace, parses dates, converts numeric fields, and removes duplicate or invalid rows.
- Uses the area-level dataset for rural/urban trends and the state-level dataset for 2020 comparisons.
- Reports the March transition separately and compares Jan–Feb 2020 with Apr–May 2020.
- Includes state-level, rural/urban, and COVID-period charts and CSV summaries.

**Finding:** the equal-weight state mean increased from **9.23%** in Jan–Feb to **22.75%** in Apr–May 2020. Across 26 states present in both periods, the median state-level change was **+11.20 percentage points**.

**Data note:** despite its filename, the state dataset ends in **October 2020**. State coverage varies by month, and the short observation window is not long enough to establish a recurring annual seasonal pattern. The comparison is descriptive and does not establish causation.

**Files:** [`unemployment_analysis.py`](Task%202/unemployment_analysis.py) · [`key_findings.txt`](Task%202/outputs/key_findings.txt) · [`state_unemployment_2020.png`](Task%202/outputs/state_unemployment_2020.png)

### 🚘 Task 3 — Car Price Prediction with Machine Learning

**Goal:** estimate a car's selling price from its age, present price, kilometers driven, ownership count, fuel type, seller type, and transmission.

- Creates vehicle age relative to the latest model year in the dataset (**2018**).
- One-hot encodes categorical features and scales numeric features inside a training pipeline.
- Compares a mean baseline, Ridge regression, random forest, and gradient boosting using five-fold training RMSE.
- Excludes `Car_Name` to avoid fitting high-cardinality name labels in this small dataset.

**Selected model:** gradient boosting. Holdout **MAE 0.778**, **RMSE 1.170**, **R² 0.941** (price units match the supplied dataset).

**Files:** [`car_price_prediction.py`](Task%203/car_price_prediction.py) · [`metrics.json`](Task%203/outputs/metrics.json) · [`actual_vs_predicted.png`](Task%203/outputs/actual_vs_predicted.png) · [`car_price_model.joblib`](Task%203/outputs/car_price_model.joblib)

### 📣 Task 4 — Sales Prediction using Python

**Goal:** estimate sales from advertising spend across TV, radio, and newspaper channels.

- Removes the source row-index column and checks that the predictors and target are numeric.
- Compares a mean baseline, linear regression, and random forest using five-fold training RMSE.
- Evaluates the selected model on a held-out test split.
- Fits a separate linear model to describe each channel's association with sales while accounting for the other channels.

**Selected model:** random forest. Holdout **MAE 0.651**, **RMSE 0.813**, **R² 0.979**. In the training-split linear model, radio has the largest coefficient and newspaper the smallest.

**Data note:** the supplied dataset has no time column, so this project estimates sales from advertising spend rather than making a chronological forecast. Channel coefficients show associations in this dataset, not guaranteed or causal returns.

**Files:** [`sales_prediction.py`](Task%204/sales_prediction.py) · [`metrics.json`](Task%204/outputs/metrics.json) · [`key_findings.txt`](Task%204/outputs/key_findings.txt) · [`channel_coefficients.png`](Task%204/outputs/channel_coefficients.png)

## 🧪 Methodology

- Fixed random seed (**42**) for reproducible train-test splits and cross-validation shuffles.
- Regression model selection uses five-fold cross-validation on the training split; the test split is held back for final evaluation.
- Classification uses a stratified split and macro-F1 so each Iris class contributes equally to model selection.
- Data preparation and model fitting are kept in scikit-learn pipelines where appropriate to prevent preprocessing from learning from the test data.
- Each script can be run independently from the repository root or from its own task folder.

## 🚀 Run the projects

### 1. Install Python packages

Use Python 3.10 or newer, then install the shared dependencies:

```bash
python -m pip install pandas scikit-learn matplotlib joblib
```

### 2. Run a task

From the repository root, run any script:

```bash
python "Task 1/iris_flower_classification.py"
python "Task 2/unemployment_analysis.py"
python "Task 3/car_price_prediction.py"
python "Task 4/sales_prediction.py"
```

Each script reads the dataset next to it and writes its tables, figures, metrics, and saved model (where applicable) to that task's `outputs/` folder.

## 🗂️ Repository layout

```text
.
├── assets/
│   └── codealpha-data-science-banner.gif
├── Task 1/
│   ├── Iris.csv
│   ├── iris_flower_classification.py
│   └── outputs/
├── Task 2/
│   ├── Unemployment in India.csv
│   ├── Unemployment_Rate_upto_11_2020.csv
│   ├── unemployment_analysis.py
│   └── outputs/
├── Task 3/
│   ├── car data.csv
│   ├── car_price_prediction.py
│   └── outputs/
├── Task 4/
│   ├── Advertising.csv
│   ├── sales_prediction.py
│   └── outputs/
└── README.md
```

## 📌 Results at a glance

| Task | Selected model | Evaluation |
| --- | --- | --- |
| Iris classification | Logistic regression | Accuracy **0.933** · Macro-F1 **0.933** |
| Car price prediction | Gradient boosting | MAE **0.778** · RMSE **1.170** · R² **0.941** |
| Advertising sales prediction | Random forest | MAE **0.651** · RMSE **0.813** · R² **0.979** |

> Scores are based on the supplied datasets and the fixed split seed in each script. They are project evaluation results, not guarantees of performance on new data.

---

<p align="center"><strong>Built with Python · Pandas · Scikit-learn · Matplotlib</strong><br>CodeAlpha Data Science Internship</p>
