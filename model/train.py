"""
model/train.py
--------------
Step 3 + 4 - Train classifiers with scikit-learn and track experiments with
MLflow.

What this module does
---------------------
  1. Reads the cleaned Parquet back into pandas.
  2. Splits 80/20 train/test, stratified by target.
  3. Runs TWO MLflow experiment runs with different hyperparameter sets:
       Run A - Logistic Regression  (C=0.1)
       Run B - Random Forest        (n_estimators=200, max_depth=6)
  4. For each run logs:
       * Parameters  (model type + hyperparams)
       * Metrics     (accuracy, precision, recall, F1, ROC-AUC)
       * Artifact    (confusion-matrix PNG, the fitted model)
  5. Prints a side-by-side comparison and identifies the best run by F1.

Run standalone:
    python -m model.train
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # headless - no GUI required
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# -- paths -------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_DIR  = os.path.join(BASE_DIR, "data", "processed", "heart_clean.parquet")
# MLflow 3.x dropped flat-file store; SQLite is the lightweight replacement
MLFLOW_DB    = os.path.join(BASE_DIR, "mlflow.db")
MLFLOW_URI   = f"sqlite:///{MLFLOW_DB.replace(os.sep, '/')}"
EXPERIMENT   = "heart-disease-classifier"

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal",
    "age_group", "chol_bp_ratio", "heart_effort",
]
TARGET_COL = "target"


# -- data loader -------------------------------------------------------------

def load_parquet(parquet_dir: str = PARQUET_DIR) -> pd.DataFrame:
    """
    Spark writes Parquet as a directory; find the actual .parquet file inside.
    """
    parts = glob.glob(os.path.join(parquet_dir, "*.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"No .parquet files found in {parquet_dir}.\n"
            "Run the transform step first:  python -m transform.transform"
        )
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    print(f"[train] Loaded {len(df)} rows from Parquet.")
    return df


# -- evaluation helpers ------------------------------------------------------

def _evaluate(model, X_test, y_test, run_name: str, tmp_dir: str):
    """Compute + return all metrics; save confusion-matrix PNG."""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0,
    }

    # Confusion matrix PNG
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Disease"]).plot(ax=ax)
    ax.set_title(f"Confusion Matrix - {run_name}")
    fig.tight_layout()
    img_path = os.path.join(tmp_dir, f"confusion_matrix_{run_name}.png")
    fig.savefig(img_path, dpi=100)
    plt.close(fig)

    return metrics, img_path


def _cv_score(pipeline, X_train, y_train):
    """5-fold stratified cross-validation; return mean F1."""
    scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=5, scoring="f1", n_jobs=-1,
    )
    return float(scores.mean()), float(scores.std())


# -- single MLflow run -------------------------------------------------------

def _run_experiment(
    run_name: str,
    classifier,
    params: dict,
    X_train, X_test, y_train, y_test,
    tmp_dir: str,
):
    """
    Log one MLflow run.  Returns (run_id, metrics_dict).
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    classifier),
    ])

    with mlflow.start_run(run_name=run_name) as run:
        # Cross-validation
        cv_mean, cv_std = _cv_score(pipeline, X_train, y_train)
        mlflow.log_metric("cv_f1_mean", cv_mean)
        mlflow.log_metric("cv_f1_std",  cv_std)
        print(f"  [CV]  {run_name}  -> F1 mean={cv_mean:.4f}  std={cv_std:.4f}")

        # Fit on full train set
        pipeline.fit(X_train, y_train)

        # Evaluate on held-out test set
        metrics, img_path = _evaluate(pipeline, X_test, y_test, run_name, tmp_dir)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(img_path)

        # Log the full sklearn pipeline (scaler + model together)
        # MLflow 3.x / skops: tree-based models need explicit trust declaration
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=None,
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
                "sklearn.pipeline.Pipeline",
                "sklearn.preprocessing._data.StandardScaler",
                "numpy.ndarray",
            ],
        )

        run_id = run.info.run_id

    _print_metrics(run_name, metrics, cv_mean)
    return run_id, metrics


def _print_metrics(run_name: str, metrics: dict, cv_f1: float) -> None:
    w = 50
    print(f"\n  {'-'*w}")
    print(f"  Run: {run_name}")
    print(f"  {'-'*w}")
    for k, v in metrics.items():
        print(f"  {k:<12} {v:.4f}")
    print(f"  {'cv_f1':<12} {cv_f1:.4f}")
    print(f"  {'-'*w}")


# -- orchestrator ------------------------------------------------------------

def train(parquet_dir: str = PARQUET_DIR) -> str:
    """
    Run both experiments; return the run_id of the best model (highest F1).
    """
    # Setup
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    tmp_dir = os.path.join(BASE_DIR, "data", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Load data
    df = load_parquet(parquet_dir)
    X  = df[FEATURE_COLS].values
    y  = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[train] Train={len(X_train)}, Test={len(X_test)}")

    results = {}

    # Run A: Logistic Regression
    print("\n[train] -- Run A: Logistic Regression (C=0.1) --")
    lr_params = {"model": "LogisticRegression", "C": 0.1, "max_iter": 1000}
    run_id_a, metrics_a = _run_experiment(
        run_name   = "LogReg_C0.1",
        classifier = LogisticRegression(C=0.1, max_iter=1000, random_state=42),
        params     = lr_params,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        tmp_dir    = tmp_dir,
    )
    results[run_id_a] = {"name": "LogReg_C0.1", "metrics": metrics_a}

    # Run B: Random Forest
    print("\n[train] -- Run B: Random Forest (n=200, depth=6) --")
    rf_params = {
        "model": "RandomForest", "n_estimators": 200,
        "max_depth": 6, "random_state": 42,
    }
    run_id_b, metrics_b = _run_experiment(
        run_name   = "RF_n200_depth6",
        classifier = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42
        ),
        params     = rf_params,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        tmp_dir    = tmp_dir,
    )
    results[run_id_b] = {"name": "RF_n200_depth6", "metrics": metrics_b}

    # Pick best by F1
    best_id = max(results, key=lambda rid: results[rid]["metrics"]["f1"])
    best    = results[best_id]
    print(
        f"\n[train] [OK] Best run -> '{best['name']}' "
        f"(F1={best['metrics']['f1']:.4f}, run_id={best_id[:8]}...)"
    )

    # Persist best run_id so the serving layer can load it
    run_id_file = os.path.join(BASE_DIR, "best_run_id.txt")
    with open(run_id_file, "w") as f:
        f.write(best_id)
    print(f"[train] Best run_id saved -> {run_id_file}")

    return best_id


# -- standalone --------------------------------------------------------------
if __name__ == "__main__":
    best = train()
    print(f"\n[train] Done. Best run_id: {best}")
