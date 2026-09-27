# DataPipe-ML 🫀

> **A bite-sized, end-to-end data engineering + ML pipeline** – raw CSV → PySpark
> transform → scikit-learn training → MLflow experiment tracking → FastAPI serving.
> Runnable in one command, explainable in five minutes.

---

## Project Layout

```
DataPipe-ML/
├── data/
│   ├── raw/              ← source CSV (heart_disease.csv)
│   └── processed/        ← Parquet output from the transform step
│
├── ingestion/
│   └── ingest.py         ← Step 1: Load CSV → PySpark DataFrame + validate
│
├── transform/
│   └── transform.py      ← Step 2: Clean, impute, engineer features → Parquet
│
├── model/
│   └── train.py          ← Step 3+4: Train classifiers + MLflow tracking
│
├── serving/
│   └── app.py            ← Step 5: FastAPI /predict endpoint
│
├── pipeline.py           ← Master orchestrator (runs all steps in sequence)
├── requirements.txt
└── README.md             ← you are here
```

---

## Pipeline Flow

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  data/raw/heart_disease.csv  (UCI Heart Disease dataset)        │
 └────────────────────────┬────────────────────────────────────────┘
                          │  PySpark read with explicit schema
                          ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  ingestion/ingest.py                                            │
 │  • Schema enforcement (typed StructType)                        │
 │  • Null-count validation                                        │
 │  • Descriptive statistics                                       │
 └────────────────────────┬────────────────────────────────────────┘
                          │  Spark DataFrame
                          ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  transform/transform.py                                         │
 │  • Drop rows with null target                                   │
 │  • Median imputation for remaining nulls                        │
 │  • Feature engineering:                                         │
 │      age_group     = bucket(age, [<40, <55, <65, ≥65])         │
 │      chol_bp_ratio = cholesterol / resting_bp                  │
 │      heart_effort  = max_hr / (1 + ST_depression)              │
 │  • Coalesce → write Parquet                                     │
 └────────────────────────┬────────────────────────────────────────┘
                          │  data/processed/heart_clean.parquet
                          ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  model/train.py                                                 │
 │  • 80/20 stratified train/test split                            │
 │  • 5-fold cross-validation                                      │
 │  ┌──────────────────────────────────────────────────────────┐   │
 │  │  MLflow Run A – LogisticRegression (C=0.1)               │   │
 │  │  MLflow Run B – RandomForest (n=200, depth=6)            │   │
 │  └──────────────────────────────────────────────────────────┘   │
 │  • Logs: params, metrics, confusion-matrix PNG, model artifact  │
 │  • Saves best run_id → best_run_id.txt                          │
 └────────────────────────┬────────────────────────────────────────┘
                          │  mlruns/  (MLflow tracking store)
                          ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  serving/app.py  (FastAPI)                                      │
 │  • Loads best model from MLflow on startup                      │
 │  • POST /predict → prediction + probability + engineered feats  │
 │  • GET  /model-info → experiment metadata                       │
 └─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
# From the DataPipe-ML/ directory
python pipeline.py
```

This will:
- Load and validate the CSV with PySpark
- Clean and feature-engineer → write Parquet
- Train two models with MLflow tracking
- Print the best run ID

### 3. Launch the API server

```bash
uvicorn serving.app:app --reload --port 8000
```

### 4. Make a prediction

**PowerShell:**
```powershell
$body = @{
  age=55; sex=1; cp=0; trestbps=140; chol=220
  fbs=0; restecg=1; thalach=155; exang=0; oldpeak=1.2
  slope=2; ca=0; thal=2
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" `
  -Method POST -ContentType "application/json" -Body $body
```

**curl:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":55,"sex":1,"cp":0,"trestbps":140,"chol":220,
       "fbs":0,"restecg":1,"thalach":155,"exang":0,"oldpeak":1.2,
       "slope":2,"ca":0,"thal":2}'
```

**Browser interactive docs:** http://localhost:8000/docs

### 5. View MLflow experiment UI

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
# Open http://localhost:5000
```

---

## Running Individual Steps

```bash
python pipeline.py --step ingest      # Step 1 only
python pipeline.py --step transform   # Steps 1-2
python pipeline.py --step train       # Steps 1-3
python pipeline.py                    # All steps (same as --step all)
```

Or run each module directly:

```bash
python -m ingestion.ingest
python -m transform.transform
python -m model.train
```

---

## Dataset

**UCI Heart Disease (Cleveland subset)** – 101 samples, 14 features.

| Column     | Description                           | Type    |
|------------|---------------------------------------|---------|
| `age`      | Age in years                          | int     |
| `sex`      | 1 = male, 0 = female                  | int     |
| `cp`       | Chest pain type (0–3)                 | int     |
| `trestbps` | Resting blood pressure (mmHg)         | int     |
| `chol`     | Serum cholesterol (mg/dl)             | int     |
| `fbs`      | Fasting blood sugar > 120 mg/dl       | int     |
| `restecg`  | Resting ECG results (0–2)             | int     |
| `thalach`  | Maximum heart rate achieved           | int     |
| `exang`    | Exercise induced angina               | int     |
| `oldpeak`  | ST depression (exercise vs rest)      | float   |
| `slope`    | Slope of peak ST segment (0–2)        | int     |
| `ca`       | # major vessels coloured (0–4)        | int     |
| `thal`     | Thalassemia type                      | int     |
| **target** | **1 = heart disease, 0 = healthy**    | **int** |

---

## Engineered Features

| Feature          | Formula                                | Rationale |
|------------------|----------------------------------------|-----------|
| `age_group`      | Ordinal bucket of age (0–3)            | Captures non-linear age risk |
| `chol_bp_ratio`  | `chol / trestbps`                      | Combined cardiovascular risk signal |
| `heart_effort`   | `thalach / (1 + oldpeak)`              | How hard the heart works under stress |

---

## What Each Metric Means

| Metric        | Formula                                          | Interpretation |
|---------------|--------------------------------------------------|----------------|
| **Accuracy**  | (TP+TN) / all                                    | % of all patients classified correctly. Misleading on imbalanced data. |
| **Precision** | TP / (TP+FP)                                     | Of all patients we *predicted* have disease, how many actually do? High precision → fewer false alarms. |
| **Recall**    | TP / (TP+FN)                                     | Of all patients who *actually have* disease, how many did we catch? High recall → fewer missed cases. |
| **F1**        | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of Precision and Recall. **Primary metric here** – balances false positives vs false negatives. |
| **ROC-AUC**   | Area under ROC curve                             | Probability that the model ranks a sick patient higher than a healthy one. 1.0 = perfect, 0.5 = random. |
| **CV F1**     | Mean F1 across 5 cross-validation folds          | Estimates how well the model generalises beyond the specific train/test split. |

### Confusion Matrix

```
              Predicted Healthy  Predicted Disease
Actual Healthy       TN                 FP
Actual Disease       FN                 TP
```

- **FP (False Positive):** Patient is healthy but model says sick → unnecessary treatment.
- **FN (False Negative):** Patient is sick but model says healthy → missed diagnosis.
- In a medical context, minimising FN (high Recall) is usually more important.

---

## MLflow Experiment Runs

| Run Name          | Model               | Key Hyperparams              |
|-------------------|---------------------|------------------------------|
| `LogReg_C0.1`     | LogisticRegression  | C=0.1, max_iter=1000         |
| `RF_n200_depth6`  | RandomForestClassifier | n_estimators=200, depth=6 |

Both runs log: parameters, accuracy/precision/recall/F1/ROC-AUC, 5-fold CV F1, confusion-matrix PNG, and the full sklearn Pipeline (StandardScaler + classifier).

---

## API Reference

### `POST /predict`

**Request body:**
```json
{
  "age": 55, "sex": 1, "cp": 0, "trestbps": 140,
  "chol": 220, "fbs": 0, "restecg": 1, "thalach": 155,
  "exang": 0, "oldpeak": 1.2, "slope": 2, "ca": 0, "thal": 2
}
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Heart Disease",
  "probability_disease": 0.7823,
  "probability_healthy": 0.2177,
  "model_run_id": "abc123...",
  "features_used": {
    "age": 55, "...",
    "age_group": 2,
    "chol_bp_ratio": 1.5714,
    "heart_effort": 129.1667
  }
}
```

### `GET /model-info`
Returns the loaded experiment name, run ID, parameters, and test metrics.

---

## Tech Stack

| Layer             | Technology          | Purpose |
|-------------------|---------------------|---------|
| Ingestion         | **PySpark 3.x**     | Scalable CSV load + schema validation |
| Transform         | **PySpark + pandas**| Cleaning, imputation, feature engineering |
| Storage           | **Parquet / PyArrow**| Columnar, compressed intermediate data |
| Model Training    | **scikit-learn**    | LogisticRegression, RandomForest |
| Experiment Tracking | **MLflow**        | Parameter/metric logging, model artifacts |
| Serving           | **FastAPI + uvicorn**| REST API for real-time inference |
| Validation        | **Pydantic v2**     | Request schema validation |
