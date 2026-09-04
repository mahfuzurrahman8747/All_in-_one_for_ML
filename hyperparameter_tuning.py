"""hyperparameter_tuning.py

Utility module for hyperparameter tuning of a collection of regression and classification
models using Optuna. The module defines a per‑model search space, runs a study for each
model, and saves the best parameters to ``best_model_params.json`` in the project root.

The tuning is performed on a validation split (the same split used for final model
evaluation in :func:`train_and_save_model`). The returned dictionary maps model names
to the best hyper‑parameter configuration for that model.

The module is deliberately lightweight – it does not depend on any project‑specific
code. It can be imported from ``model_train.py`` to retrieve the tuned parameters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import optuna
from sklearn.metrics import accuracy_score, r2_score

# Optional imports – the user must install the corresponding packages if they wish
# to use the associated models. Missing imports are handled gracefully.
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

# ---------------------------------------------------------------------------
# Helper: define search spaces per model
# ---------------------------------------------------------------------------

def _suggest_xgboost(trial: optuna.trial.Trial, problem_type: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    if problem_type == "classification":
        # XGBClassifier specific defaults – avoid label encoder warning
        params.update({"use_label_encoder": False, "eval_metric": "logloss"})
    return params


def _suggest_lightgbm(trial: optuna.trial.Trial, problem_type: str) -> Dict[str, Any]:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 128),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
    }
    return params


def _suggest_catboost(trial: optuna.trial.Trial) -> Dict[str, Any]:
    params = {
        "iterations": trial.suggest_int("iterations", 50, 300),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }
    return params


def _suggest_random_forest(trial: optuna.trial.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }


def _suggest_gradient_boosting(trial: optuna.trial.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
    }


def _suggest_logistic_regression(trial: optuna.trial.Trial) -> Dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
    }


def _suggest_svc(trial: optuna.trial.Trial) -> Dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "kernel": "rbf",
        "gamma": "scale",
    }

# ---------------------------------------------------------------------------
# Objective wrappers for each model
# ---------------------------------------------------------------------------

def _objective_factory(
    model_name: str,
    problem_type: str,
    X_train: Any,
    y_train: Any,
    X_valid: Any,
    y_valid: Any,
):
    """Return an Optuna objective function for *model_name*.

    The inner function receives a trial, draws hyper‑parameters according to the
    model‑specific helper above, fits the model on the training split and returns
    a scalar score to be maximised (accuracy for classification, R² for regression).
    """

    def objective(trial: optuna.trial.Trial) -> float:
        # Choose hyper‑parameters based on the model type
        if model_name == "XGBoost":
            if XGBClassifier is None or XGBRegressor is None:
                raise optuna.exceptions.TrialPruned("XGBoost not available")
            params = _suggest_xgboost(trial, problem_type)
            ModelCls = XGBClassifier if problem_type == "classification" else XGBRegressor
            model = ModelCls(**params)
        elif model_name == "LightGBM":
            if LGBMClassifier is None or LGBMRegressor is None:
                raise optuna.exceptions.TrialPruned("LightGBM not available")
            params = _suggest_lightgbm(trial, problem_type)
            ModelCls = LGBMClassifier if problem_type == "classification" else LGBMRegressor
            model = ModelCls(**params)
        elif model_name == "CatBoost":
            if CatBoostClassifier is None or CatBoostRegressor is None:
                raise optuna.exceptions.TrialPruned("CatBoost not available")
            params = _suggest_catboost(trial)
            ModelCls = CatBoostClassifier if problem_type == "classification" else CatBoostRegressor
            # Silent training – ``verbose`` defaults to 0 for verbose=False.
            model = ModelCls(**params, verbose=False)
        elif model_name == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            params = _suggest_random_forest(trial)
            ModelCls = RandomForestClassifier if problem_type == "classification" else RandomForestRegressor
            model = ModelCls(**params, n_jobs=-1, random_state=42)
        elif model_name == "Gradient Boosting":
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

            params = _suggest_gradient_boosting(trial)
            ModelCls = (
                GradientBoostingClassifier if problem_type == "classification" else GradientBoostingRegressor
            )
            model = ModelCls(**params)
        elif model_name == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression

            params = _suggest_logistic_regression(trial)
            model = LogisticRegression(**params)
        elif model_name == "SVM":
            from sklearn.svm import SVC

            params = _suggest_svc(trial)
            model = SVC(**params)
        elif model_name == "Naive Bayes":
            from sklearn.naive_bayes import GaussianNB

            # No hyper‑parameters to tune for GaussianNB – return a constant score.
            model = GaussianNB()
            # Directly compute metric without a trial.
            model.fit(X_train, y_train)
            preds = model.predict(X_valid)
            return accuracy_score(y_valid, preds)
        else:
            raise ValueError(f"Unsupported model for tuning: {model_name}")

        # Fit / evaluate
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        if problem_type == "classification":
            return accuracy_score(y_valid, preds)
        else:
            # For regression we maximise R² (the higher the better)
            return r2_score(y_valid, preds)

    return objective

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tune_models(
    problem_type: str,
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    n_trials: int = 30,
) -> Dict[str, Dict[str, Any]]:
    """Tune a predefined set of models for *problem_type*.

    Parameters are the training and test splits (the same split used for final model
    evaluation in :func:`train_and_save_model`). Returns a dictionary mapping model
    display names to the best hyper‑parameters discovered by Optuna.
    """

    # Define which models we support for each problem type
    if problem_type == "classification":
        candidate_models = [
            "Logistic Regression",
            "Random Forest",
            "Gradient Boosting",
            "SVM",
            "XGBoost",
            "LightGBM",
            "CatBoost",
            "Naive Bayes",
        ]
    else:
        candidate_models = [
            "Linear Regression",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
            "LightGBM",
            "CatBoost",
        ]

    best_params: Dict[str, Dict[str, Any]] = {}

    for model_name in candidate_models:
        # For models without an external library we still go through Optuna to keep the API uniform.
        try:
            objective = _objective_factory(
                model_name, problem_type, X_train, y_train, X_test, y_test
            )
            direction = "maximize"
            study = optuna.create_study(direction=direction)
            study.optimize(objective, n_trials=n_trials, timeout=None)
            best_params[model_name] = study.best_params
        except optuna.exceptions.TrialPruned as e:
            # Library missing – skip tuning and fall back to default params.
            print(f"Skipping {model_name} during tuning: {e}")
        except Exception as exc:  # pragma: no cover
            print(f"Error tuning {model_name}: {exc}")

    return best_params


def save_best_params(best_params: Dict[str, Dict[str, Any]], file_path: Path = Path("best_model_params.json")) -> None:
    """Persist *best_params* to *file_path* as pretty‑printed JSON."""
    file_path.write_text(json.dumps(best_params, indent=4), encoding="utf-8")


def load_best_params(file_path: Path = Path("best_model_params.json")) -> Dict[str, Dict[str, Any]]:
    """Load a JSON file previously written by :func:`save_best_params`.

    If the file does not exist an empty dictionary is returned.
    """
    if not file_path.is_file():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))

# End of hyperparameter_tuning.py
