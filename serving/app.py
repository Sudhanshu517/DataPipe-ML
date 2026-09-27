"""
serving/app.py
──────────────
Step 5 – FastAPI service that loads the best MLflow-tracked model and exposes
a /predict endpoint.

Endpoints
---------
  GET  /            → health-check / welcome
  GET  /model-info  → which MLflow run & experiment is loaded
  POST /predict     → returns prediction + probability for a new patient

Run:
    uvicorn serving.app:app --reload --port 8000

Then test:
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"age":55,"sex":1,"cp":0,"trestbps":140,"chol":220,"fbs":0,
           "restecg":1,"thalach":155,"exang":0,"oldpeak":1.2,
           "slope":2,"ca":0,"thal":2}'
"""

import os
import glob
import logging
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_DB     = os.path.join(BASE_DIR, "mlflow.db")
MLFLOW_URI    = f"sqlite:///{MLFLOW_DB.replace(os.sep, '/')}"
EXPERIMENT    = "heart-disease-classifier"
RUN_ID_FILE   = os.path.join(BASE_DIR, "best_run_id.txt")

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal",
    "age_group", "chol_bp_ratio", "heart_effort",
]


# ── request schema ────────────────────────────────────────────────────────────

class PatientFeatures(BaseModel):
    """Raw patient vitals – the API derives engineered features automatically."""

    age:      int   = Field(..., ge=1,   le=120, description="Age in years")
    sex:      int   = Field(..., ge=0,   le=1,   description="Sex (1=male, 0=female)")
    cp:       int   = Field(..., ge=0,   le=3,   description="Chest pain type (0–3)")
    trestbps: int   = Field(..., ge=50,  le=250, description="Resting blood pressure (mmHg)")
    chol:     int   = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs:      int   = Field(..., ge=0,   le=1,   description="Fasting blood sugar > 120 mg/dl")
    restecg:  int   = Field(..., ge=0,   le=2,   description="Resting ECG results (0–2)")
    thalach:  int   = Field(..., ge=50,  le=250, description="Maximum heart rate achieved")
    exang:    int   = Field(..., ge=0,   le=1,   description="Exercise induced angina (1=yes)")
    oldpeak:  float = Field(..., ge=0.0, le=10.0,description="ST depression (exercise vs rest)")
    slope:    int   = Field(..., ge=0,   le=2,   description="Slope of peak ST segment (0–2)")
    ca:       int   = Field(..., ge=0,   le=4,   description="# major vessels coloured (0–4)")
    thal:     int   = Field(..., ge=1,   le=7,   description="Thalassemia type")

    @field_validator("ca")
    @classmethod
    def ca_range(cls, v):
        if v not in range(5):
            raise ValueError("ca must be 0–4")
        return v


class PredictionResponse(BaseModel):
    prediction:        int
    label:             str
    probability_disease: float
    probability_healthy: float
    model_run_id:      str
    features_used:     dict


# ── feature derivation (mirrors transform step) ───────────────────────────────

def derive_features(p: PatientFeatures) -> np.ndarray:
    """Compute the 3 engineered features and assemble the feature vector."""
    age_group     = (0 if p.age < 40 else 1 if p.age < 55 else 2 if p.age < 65 else 3)
    chol_bp_ratio = p.chol / p.trestbps
    heart_effort  = p.thalach / (1.0 + p.oldpeak)

    raw = [
        p.age, p.sex, p.cp, p.trestbps, p.chol,
        p.fbs, p.restecg, p.thalach, p.exang, p.oldpeak,
        p.slope, p.ca, p.thal,
        age_group, chol_bp_ratio, heart_effort,
    ]
    return np.array(raw, dtype=float).reshape(1, -1)


# ── model loader ──────────────────────────────────────────────────────────────

_state: dict = {}   # holds model + metadata after startup


def _load_best_model():
    """
    Load the best MLflow model from the saved run_id file.
    Falls back to scanning runs and picking the highest F1.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)   # sqlite:///mlflow.db
    client = mlflow.tracking.MlflowClient()

    # Try reading persisted best run_id
    if os.path.exists(RUN_ID_FILE):
        with open(RUN_ID_FILE) as f:
            run_id = f.read().strip()
        logger.info(f"Loading model for run_id={run_id[:8]}…")
    else:
        # Fallback: find experiment and pick best F1 run
        exp = client.get_experiment_by_name(EXPERIMENT)
        if exp is None:
            raise RuntimeError(
                "No MLflow experiment found. Run 'python pipeline.py' first."
            )
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.f1 DESC"],
            max_results=1,
        )
        if not runs:
            raise RuntimeError("No runs found. Run 'python pipeline.py' first.")
        run_id = runs[0].info.run_id

    model_uri = f"runs:/{run_id}/model"
    model     = mlflow.sklearn.load_model(model_uri)
    run       = client.get_run(run_id)

    return model, run_id, run.data.params, run.data.metrics


# ── lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once on startup, release on shutdown."""
    logger.info("── DataPipe-ML Serving Layer starting …")
    try:
        model, run_id, params, metrics = _load_best_model()
        _state["model"]   = model
        _state["run_id"]  = run_id
        _state["params"]  = params
        _state["metrics"] = metrics
        logger.info(
            f"✓ Model loaded  run_id={run_id[:8]}…  "
            f"F1={metrics.get('f1', 'n/a')}"
        )
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        _state["model"] = None
        _state["error"] = str(e)
    yield
    logger.info("── DataPipe-ML Serving Layer shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "DataPipe-ML  |  Heart Disease Predictor",
    description = (
        "End-to-end ML serving layer. Loads the best scikit-learn model "
        "tracked in MLflow and returns a heart-disease prediction."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "DataPipe-ML",
        "status":  "ok" if _state.get("model") else "model_not_loaded",
        "docs":    "/docs",
    }


@app.get("/model-info", tags=["Health"])
def model_info():
    if not _state.get("model"):
        raise HTTPException(503, detail=_state.get("error", "Model not loaded"))
    return {
        "experiment":  EXPERIMENT,
        "run_id":      _state["run_id"],
        "parameters":  _state["params"],
        "test_metrics": _state["metrics"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(patient: PatientFeatures):
    """
    Accepts raw patient features, derives engineered features, and returns:
      - prediction      : 1 (disease) or 0 (healthy)
      - label           : human-readable label
      - probability_*   : class probabilities from the model
    """
    if not _state.get("model"):
        raise HTTPException(503, detail=_state.get("error", "Model not loaded"))

    model = _state["model"]

    # Build feature vector (includes engineered features)
    X = derive_features(patient)

    # Predict
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    engineered = {
        "age_group":     int(0 if patient.age < 40 else 1 if patient.age < 55 else 2 if patient.age < 65 else 3),
        "chol_bp_ratio": round(patient.chol / patient.trestbps, 4),
        "heart_effort":  round(patient.thalach / (1.0 + patient.oldpeak), 4),
    }

    return PredictionResponse(
        prediction           = pred,
        label                = "Heart Disease" if pred == 1 else "Healthy",
        probability_disease  = round(float(proba[1]), 4),
        probability_healthy  = round(float(proba[0]), 4),
        model_run_id         = _state["run_id"],
        features_used        = {**patient.model_dump(), **engineered},
    )
