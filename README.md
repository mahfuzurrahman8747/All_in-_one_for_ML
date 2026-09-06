# All-in-One for Machine Learning (AutoML Toolkit)

![alt text](<Screenshot (119).png>)

**All-in-One for ML** is a lightweight, end‑to‑end AutoML toolkit that simplifies the process of training, tuning, and deploying machine‑learning models on tabular data. It provides a Streamlit web interface where users can upload a dataset, select a target column, and instantly obtain a trained model with performance metrics and a downloadable model artifact.

> **Why this project?**
> - No need to write boilerplate code for data preprocessing, model selection, or hyper‑parameter tuning. 
> - Works out‑of‑the‑box for both regression and classification problems.
> - Provides transparent results and a ready‑to‑use model that can be integrated into downstream applications.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Data Requirements](#data-requirements)
- [How It Works](#how-it-works)
-   - [Data Loading & Cleaning](#data-loading--cleaning)
-   - [Problem Type Detection](#problem-type-detection)
-   - [Model Candidates & Hyper‑Parameter Tuning](#model-candidates--hyper‑parameter-tuning)
-   - [Model Evaluation & Selection](#model-evaluation--selection)
-   - [Model Persistence & Download](#model-persistence--download)
- [Streamlit App Usage](#streamlit-app-usage)
-   - [Screenshot Placeholder](#screenshot-placeholder)
- [Running the Training Pipeline Programmatically](#running-the-training-pipeline-programmatically)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Upload any tabular dataset** (`.csv` or `.xlsx`).
- Automatic **target column selection** and **basic cleaning** (missing values, duplicates, outlier removal, identifier column drop, one‑hot encoding).
- **Problem type inference** (classification vs. regression) based on target column characteristics.
- **Model zoo** with popular algorithms:
  - Linear / Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Machines (SVM / SVR)
  - XGBoost, LightGBM, CatBoost (optional – installed on demand)
  - Naïve Bayes (classification only)
- **Optuna‑powered hyper‑parameter tuning** for each model (default 30 trials).
- **Automatic model selection** using the best primary metric (Accuracy for classification, R² for regression).
- **Model bundling** (`best_model.joblib`) that includes the trained estimator, scaler, feature‑column list, and target‑encoding information.
- **Result dashboard** with key metrics (Accuracy, MAE, RMSE, R², etc.) displayed in Streamlit.
- **Downloadable model artifact** and ready‑to‑paste inference code snippet.
- **Extensible design** – you can call `train_and_save_model` from Python scripts or notebooks.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your‑username/All-in-One-ML.git
cd All-in-One-ML

# (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Optional dependencies for extra models (install only if you need them)
# XGBoost
pip install xgboost
# LightGBM
pip install lightgbm
# CatBoost
pip install catboost
```

> **Note:** The `requirements.txt` contains `streamlit`, `pandas`, `scikit‑learn`, `joblib`, `optuna`, and other essential packages.

---

## Quick Start

1. **Launch the Streamlit UI**
   ```bash
   streamlit run app.py
   ```

2. In the web UI:
   - Upload your dataset (CSV or Excel).
   - Choose the **target column** you want to predict.
   - Click **"Train model"**.
   - Wait a few seconds while the pipeline runs (data cleaning → tuning → model selection).
   - Review the metrics that appear on the page.
   - Download `best_model.joblib` if you wish to use the model elsewhere.

3. **Run programmatically** (optional):
   ```python
   from model_train import train_and_save_model
   summary = train_and_save_model(data_path=Path('uploads/my_data.csv'),
                                 target_col='SalePrice')
   print(summary)
   ```

---

## Project Structure

```
All_in-One-ML/
├─ app.py                     # Streamlit front‑end
├─ model_train.py             # Core training pipeline
├─ hyperparameter_tuning.py   # Optuna tuning helpers
├─ best_model_params.json    # Saved optimal hyper‑parameters (auto‑generated)
├─ best_model_results.json   # Summary of the best model (auto‑generated)
├─ model_parameters/          # Directory containing best_model.joblib
│   └─ best_model.joblib
├─ uploads/                   # Uploaded datasets (runtime only)
├─ requirements.txt
├─ README.md                  # <--- This file
└─ ... (optional notebooks, tests, etc.)
```

---

## Data Requirements

- **Tabular data** stored as CSV (`.csv`) or Excel (`.xlsx`).
- The **target column** must be present and contain either numeric values (regression) or categorical values (classification). If the column has ≤ 10 unique values or is of type `object`, the pipeline treats it as a classification problem.
- Columns that look like identifiers (e.g., `id`, `name`, `email`, `phone`) are automatically removed during preprocessing.
- Missing values and duplicate rows are dropped; extreme numeric outliers (above the 99th percentile) are clipped out.

---

## How It Works

### Data Loading & Cleaning
The `model_train._load_data` function reads CSV, Excel, or Parquet files into a `pandas.DataFrame`. The `_clean_data` helper then:
1. Drops rows with missing values and duplicates.
2. Removes extreme outliers (99th percentile).
3. Strips identifier‑like columns.
4. One‑hot encodes any remaining categorical columns.

### Problem Type Detection
`_determine_problem_type` uses a simple heuristic:
- If the target column is of type `object` **or** has ≤ 10 unique values → **classification**.
- Otherwise → **regression**.

### Model Candidates & Hyper‑Parameter Tuning
A predetermined list of models is instantiated for the detected problem type. For each model, Optuna searches a model‑specific hyper‑parameter space defined in `hyperparameter_tuning.py`. The default budget is **30 trials per model** (customizable via `n_trials`).

### Model Evaluation & Selection
All models are trained on an 80 %/20 % train‑test split (stratified for classification). Predictions for the test set are stored in a temporary DataFrame and fed to the `metrics` function, which computes:
- Accuracy (or a surrogate based on MAPE for regression)
- MAE, RMSE, MAPE, R²
The primary metric (Accuracy for classification, R² for regression) determines the **best model**.

### Model Persistence & Download
The chosen model, the fitted `StandardScaler`, feature‑column names, target information, and (for classification) the class mapping are bundled into a dictionary and saved as `model_parameters/best_model.joblib` using `joblib.dump`. A lightweight JSON summary (`best_model_results.json`) is also written for the Streamlit UI.

---

## Streamlit App Usage

The Streamlit front‑end (`app.py`) orchestrates the workflow:
- **File uploader** → stores the file under `uploads/`.
- **Target column selector** → populates a dropdown based on the uploaded data.
- **"Train model" button** → calls `train_and_save_model`.
- **Result cards** → display the latest model’s name, problem type, target column, and metrics.
- **Download button** → provides the serialized `best_model.joblib`.
- **Usage snippet** → a collapsible section shows a ready‑to‑copy code block for loading the model and making predictions.

### Screenshot Placeholder
Below is a placeholder for a screenshot of the Streamlit UI. Replace the `path/to/screenshot.png` with an actual image file when you capture one.

```markdown
![Streamlit App Screenshot](assets/streamlit_screenshot.png)
```

---

## Running the Training Pipeline Programmatically

If you prefer to use the library without the UI, import and call the helper directly:

```python
from pathlib import Path
from model_train import train_and_save_model

# Example – train on a CSV file
summary = train_and_save_model(
    data_path=Path('uploads/your_data.csv'),
    target_col='TargetColumnName',
    output_dir=Path('model_parameters')
)
print('Training complete. Summary:', summary)
```

The returned `summary` dictionary contains the same fields that the Streamlit app consumes:
- `model_name`
- `problem_type`
- `target_column`
- `metrics` (a dict with Accuracy, MAE, RMSE, etc.)

---

## Dependencies

| Package                | Purpose                                    |
|------------------------|--------------------------------------------|
| `streamlit`            | Interactive web UI                         |
| `pandas`                | Data manipulation and I/O                  |
| `numpy`                | Numerical operations                       |
| `scikit-learn`         | Core ML models, preprocessing, metrics      |
| `optuna`               | Hyper‑parameter optimization                |
| `joblib`               | Model serialization                         |
| `xgboost` *(optional)*| Gradient‑boosted trees (fast, accurate)    |
| `lightgbm` *(optional)*| Gradient‑boosted trees (lightweight)       |
| `catboost` *(optional)*| Gradient‑boosted trees (GPU‑ready)        |

Install optional packages only if you intend to use the corresponding models.

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

1. Fork the repository.
2. Create a feature branch (`git checkout -b feat/awesome-feature`).
3. Ensure the code follows existing style conventions and passes any tests.
4. Open a Pull Request with a clear description of the change.

---

## License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

*Happy modeling!*