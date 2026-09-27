"""
pipeline.py
-----------
Master orchestrator - runs the complete DataPipe-ML pipeline in sequence:

    1. Ingestion   (PySpark/pandas)   -> raw CSV -> DataFrame
    2. Transform   (pandas)           -> clean + feature-engineer -> Parquet
    3. Train       (sklearn)          -> 2 MLflow runs -> best model saved
    (4. Serve is launched separately via uvicorn)

Usage:
    python pipeline.py              # full pipeline
    python pipeline.py --step ingest
    python pipeline.py --step transform
    python pipeline.py --step train
"""

import sys
import time
import argparse
import logging

# Force UTF-8 on Windows so Unicode log chars render correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger("pipeline")


def _banner(text: str) -> None:
    bar = "=" * 62
    log.info(bar)
    log.info(f"  {text}")
    log.info(bar)


def run_ingest(spark=None):
    _banner("STEP 1 - INGESTION")
    from ingestion.ingest import get_spark, ingest
    if spark is None:
        spark = get_spark()
        if hasattr(spark, 'sparkContext'):
            spark.sparkContext.setLogLevel("ERROR")
    t0 = time.time()
    df = ingest(spark)
    log.info(f"Ingestion done in {time.time()-t0:.1f}s")
    return spark, df


def run_transform(spark, raw_df):
    _banner("STEP 2 - TRANSFORM")
    from transform.transform import transform
    t0 = time.time()
    clean_df = transform(raw_df)
    log.info(f"Transform done in {time.time()-t0:.1f}s")
    return clean_df


def run_train():
    _banner("STEP 3 - TRAIN + MLFLOW TRACKING")
    from model.train import train
    t0 = time.time()
    best_id = train()
    log.info(f"Training done in {time.time()-t0:.1f}s")
    return best_id


def main():
    parser = argparse.ArgumentParser(description="DataPipe-ML pipeline orchestrator")
    parser.add_argument(
        "--step",
        choices=["ingest", "transform", "train", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    args = parser.parse_args()

    t_start = time.time()

    if args.step in ("ingest", "all"):
        spark, raw_df = run_ingest()

    if args.step in ("transform", "all"):
        if args.step == "transform":
            spark, raw_df = run_ingest()
        clean_df = run_transform(spark, raw_df)

    if args.step in ("train", "all"):
        best_id = run_train()

    if args.step == "all":
        spark.stop()

    _banner(f"Pipeline finished in {time.time()-t_start:.1f}s [OK]")

    if args.step in ("train", "all"):
        log.info("")
        log.info("  Start the API server:")
        log.info("    uvicorn serving.app:app --reload --port 8000")
        log.info("")
        log.info("  Test prediction (PowerShell):")
        log.info(
            '    Invoke-RestMethod -Uri "http://localhost:8000/predict" '
            '-Method POST -ContentType "application/json" '
            '-Body \'{"age":55,"sex":1,"cp":0,"trestbps":140,"chol":220,'
            '"fbs":0,"restecg":1,"thalach":155,"exang":0,"oldpeak":1.2,'
            '"slope":2,"ca":0,"thal":2}\''
        )


if __name__ == "__main__":
    main()
