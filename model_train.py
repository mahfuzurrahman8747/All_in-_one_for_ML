'''model_train.py

Utility module that encapsulates the model training pipeline previously
implemented in ``EDA.ipynb``.  The :func:`train_and_save_model` function
loads a dataset, performs basic cleaning and encoding, splits the data, trains
a handful of candidate models (regression or classification depending on the
target column), selects the best model based on a simple metric, and saves
both the model bundle (``best_model.joblib``) and a JSON summary containing
the chosen model name, problem type, target column and its performance
metrics.

The function is deliberately side‑effect‑heavy – it writes files in the
project root because the existing ``app.py`` expects the files ``model_parameters/best_model.joblib``
and ``best_model_results.json`` to be present.  It returns the same summary
dictionary that gets written to ``best_model_results.json`` so callers can
use the result directly if they wish.
'''

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
# Additional model imports
from sklearn.naive_bayes import GaussianNB

# Optional third‑party models – import lazily to avoid hard failures if the package is missing.
try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover
    XGBClassifier = XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover
    LGBMClassifier = LGBMRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostClassifier = CatBoostRegressor = None

# Hyperparameter‑tuning helpers
from hyperparameter_tuning import tune_models, save_best_params, load_best_params


def _load_data(data_path: Path) -> pd.DataFrame:
    """Load ``data_path`` as a ``pandas.DataFrame``.

    Supports CSV, Excel (``.xlsx``/``.xls``) and Parquet files.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(data_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(data_path)
    # Fall back to parquet for any other extension.
    return pd.read_parquet(data_path)


def _clean_data(df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
    """Perform the same cleaning steps that the notebook applied.

    * Drop rows with missing values.
    * Drop duplicate rows.
    * Remove extreme outliers (values above the 99th percentile) for
      numeric columns.
    * Remove identifier‑like columns (``id``, ``name``, ``email`` etc.).
    * One‑hot encode remaining object/category columns.
    """
    # Drop missing and duplicate rows.
    df = df.dropna().drop_duplicates()

    # Remove outliers beyond the 99th percentile for numeric columns.
    numeric_columns = df.select_dtypes(include="number").columns
    if len(numeric_columns) > 0:
        outlier_mask = (df[numeric_columns] <= df[numeric_columns].quantile(0.99)).all(axis=1)
        df = df.loc[outlier_mask].copy()

    # Remove likely identifier columns.
    identifier_pattern = re.compile(
        r"(^|[_\s-])(id|name|email|phone|phone_num|phone_number)([_\s-]|$)",
        re.IGNORECASE,
    )
    columns_to_remove = [
        col
        for col in df.columns
        if col != target_col and identifier_pattern.search(str(col))
    ]
    df = df.drop(columns=columns_to_remove, errors="ignore")

    # One‑hot encode any remaining object or categorical columns.
    categorical_columns = [
        col
        for col in df.select_dtypes(include=["object", "str", "category"]).columns
        if col != target_col
    ]
    if len(categorical_columns) > 0:
        df = pd.get_dummies(df, columns=categorical_columns, dtype=int)

    return df


def _determine_problem_type(y: pd.Series) -> str:
    """Return ``"classification"`` or ``"regression"``.

    The notebook used the heuristic that a column with ``object`` dtype or
    ``<= 10`` unique values should be treated as a classification problem.
    """
    if y.dtype == "object" or y.nunique() <= 10:
        return "classification"
    return "regression"


def _instantiate_models(problem_type: str, best_params: Dict[str, Any]) -> Dict[str, Any]:
    """Instantiate candidate models, applying hyper‑parameters from *best_params*.

    The function returns a mapping from a human‑readable model name to an unfitted
    estimator instance. Models that depend on optional third‑party libraries are
    included only if the import succeeded (the corresponding class is not ``None``).
    """
    if problem_type == "regression":
        constructors = {
            "Linear Regression": LinearRegression,
            "Random Forest": RandomForestRegressor,
            "Gradient Boosting": GradientBoostingRegressor,
            "XGBoost": XGBRegressor,
            "LightGBM": LGBMRegressor,
            "CatBoost": CatBoostRegressor,
        }
    else:
        constructors = {
            "Logistic Regression": LogisticRegression,
            "Random Forest": RandomForestClassifier,
            "Gradient Boosting": GradientBoostingClassifier,
            "SVM": SVC,
            "XGBoost": XGBClassifier,
            "LightGBM": LGBMClassifier,
            "CatBoost": CatBoostClassifier,
            "Naive Bayes": GaussianNB,
        }
    models = {}
    best_params = best_params or {}
    for name, ctor in constructors.items():
        if ctor is None:
            # Skip models whose optional dependency is not installed.
            continue
        params = best_params.get(name) or {}
        models[name] = ctor(**params)
    return models

def metrics(name, y_true, y_pred, problem_type="regression"):
    """Calculate regression/classification metrics for a model.

    Returns a dict with model name and rounded metric values.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Avoid division by zero for MAPE – replace zeros with a small epsilon.
    y_true_numeric = y_true.astype(float)
    y_pred_numeric = y_pred.astype(float)
    y_true_safe = np.where(y_true_numeric == 0, np.finfo(float).eps, y_true_numeric)
    mape = np.mean(np.abs((y_true_numeric - y_pred_numeric) / y_true_safe)) * 100
    r2 = r2_score(y_true_numeric, y_pred_numeric)
    if problem_type == "classification":
        acc = accuracy_score(y_true, y_pred) * 100
    else:
        # Accuracy is not defined for continuous regression targets.
        acc = max(0.0, 1.0 - mape / 100.0) * 100
    return {
        "model": name,
        "Accuracy": round(acc, 3),
        "MAE": round(mae, 1),
        "RMSE": round(rmse, 1),
        "MAPE": round(mape, 3),
        "R2": round(r2, 4),
    }


def _evaluate_models(
    problem_type: str,
    result_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create a DataFrame with the performance of each model using the custom metrics.

    Returns a DataFrame sorted by the primary metric (Accuracy for classification,
    R2 for regression).
    """
    y_true = result_df["actual"]
    model_names = [col for col in result_df.columns if col != "actual"]
    records = [
        metrics(name, y_true, result_df[name], problem_type)
        for name in model_names
    ]
    metrics_df = pd.DataFrame(records)
    sort_col = "Accuracy" if problem_type == "classification" else "R2"
    return metrics_df.sort_values(sort_col, ascending=False)


def train_and_save_model(
    data_path: Path,
    target_col: str,
    output_dir: Path = Path("model_parameters"),
) -> Dict[str, Any]:
    """Train candidate models on ``data_path`` and persist the best one.

    Parameters
    ----------
    data_path:
        Path to the uploaded data file (CSV, Excel or Parquet).
    target_col:
        Column name that should be predicted.
    output_dir:
        Directory where ``best_model.joblib`` will be written.  The directory
        is created if it does not exist.

    Returns
    -------
    dict
        A summary containing the selected model name, problem type, target column
        and the metric dictionary for the chosen model.  The same dictionary
        is also written to ``best_model_results.json`` in the project root.
    """
    # ---------------------------------------------------------------------
    # 1. Load raw data
    # ---------------------------------------------------------------------
    df = _load_data(data_path)

    # ---------------------------------------------------------------------
    if target_col not in df.columns:
        raise ValueError(
            f"Target column {target_col!r} not found in the uploaded data. "
            f"Available columns: {list(df.columns)}"
        )

    target_classes = None

    # 2. Basic cleaning / feature engineering (mirrors the notebook).
    # Keep the selected target intact so categorical targets are not one-hot encoded.
    # ---------------------------------------------------------------------
    df = _clean_data(df, target_col=target_col)

    # ---------------------------------------------------------------------
    # 3. Separate target and features
    # ---------------------------------------------------------------------
    if target_col not in df.columns:
        raise ValueError(
        f"Target column {target_col!r} not found. "
        f"Available columns: {list(df.columns)}"
    )

# --------------------------------------------------
# Encode target column
# --------------------------------------------------
    if target_col not in df.columns:
        raise ValueError(
        f"Target column {target_col!r} not found. "
        f"Available columns: {list(df.columns)}"
    )

# --------------------------------------------------
# Encode target column
# --------------------------------------------------
    target_dtype = df[target_col].dtype

    if (
        pd.api.types.is_object_dtype(target_dtype)
        or pd.api.types.is_string_dtype(target_dtype)
        or pd.api.types.is_categorical_dtype(target_dtype)
    ):
        # Get unique classes, ignoring missing values
        target_classes = df[target_col].dropna().unique().tolist()

        # Create mapping: class name -> integer
        target_mapping = {
            value: index for index, value in enumerate(target_classes)
        }

        # IMPORTANT: assign encoded values back to dataframe
        df[target_col] = df[target_col].map(target_mapping)

    # Convert target to numeric explicitly.
    df[target_col] = pd.to_numeric(df[target_col], errors="raise")

    print(f"Target column '{target_col}' encoded. Classes: {target_classes}")
    print(f"First 5 rows of the DataFrame after encoding:\n{df[target_col].head(5)}")
    # Create X and y AFTER encoding
    y = df[target_col].astype(int) if target_classes is not None else df[target_col]
    X = df.drop(columns=[target_col])

    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    # ---------------------------------------------------------------------
    # 4. Determine problem type (regression vs classification)
    # ---------------------------------------------------------------------
    problem_type = _determine_problem_type(y)

    # ---------------------------------------------------------------------
    # 5. Train / test split (stratify for classification)
    # ---------------------------------------------------------------------
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y if problem_type == "classification" else None,
    )

    # ---------------------------------------------------------------------
    # 6. Feature scaling – ``StandardScaler`` is applied to **both** train and
    #    test sets so that downstream models that are sensitive to feature scale
    #    (e.g. SVR/SVM) behave as expected.
    # ---------------------------------------------------------------------
    scaler = StandardScaler()
    x_train = pd.DataFrame(
        scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index
    )
    x_test = pd.DataFrame(
        scaler.transform(x_test), columns=x_test.columns, index=x_test.index
    )

    # ---------------------------------------------------------------------
    # 7. Hyperparameter tuning and model instantiation
    # ---------------------------------------------------------------------
    params_path = Path("best_model_params.json")
    if params_path.is_file():
        best_params = load_best_params(params_path)
    else:
        # Use the same train/validation split for tuning
        best_params = tune_models(
            problem_type,
            x_train,
            x_test,
            y_train,
            y_test,
            n_trials=30,
        )
        save_best_params(best_params, params_path)

    models = _instantiate_models(problem_type, best_params)

    # ---------------------------------------------------------------------
    # 8. Fit each model and collect predictions
    # ---------------------------------------------------------------------
    result_df = pd.DataFrame({"actual": y_test.to_numpy()}, index=y_test.index)
    for name, estimator in models.items():
        estimator.fit(x_train, y_train)
        result_df[name] = estimator.predict(x_test)

    # ---------------------------------------------------------------------
    # 9. Evaluate – pick the best model according to the primary metric.
    # ---------------------------------------------------------------------
    metrics_df = _evaluate_models(problem_type, result_df)
    best_model_name = metrics_df.iloc[0]["model"]
    best_model = models[best_model_name]

    # ---------------------------------------------------------------------
    # 10. Persist the best model bundle (including scaler and feature names)
    # ---------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "model_name": best_model_name,
        "model": best_model,
        "scaler": scaler,
        "feature_columns": list(x_train.columns),
        "target_column": target_col,
        "problem_type": problem_type,
        "target_classes": target_classes,
    }
    model_path = output_dir / "best_model.joblib"
    joblib.dump(model_bundle, model_path)

    # ---------------------------------------------------------------------
    # 11. Create the JSON summary that ``app.py`` consumes.
    # ---------------------------------------------------------------------
    best_metrics = metrics_df.loc[metrics_df["model"] == best_model_name].iloc[0].to_dict()
    summary = {
        "model_name": best_model_name,
        "problem_type": problem_type,
        "target_column": target_col,
        "metrics": best_metrics,
    }
    results_path = Path("best_model_results.json")
    results_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    return summary


# The module deliberately does *not* execute any code on import – callers must
# invoke :func:`train_and_save_model` explicitly.
if __name__ == "__main__":
    file_path = Path("uploads/Test.csv")
    target_col = "Spending_Score"
    train_and_save_model(data_path=file_path, target_col=target_col)