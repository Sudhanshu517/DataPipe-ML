"""
transform/transform.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Step 2 â€“ Clean the raw DataFrame and engineer features.

Works with both a real PySpark DataFrame and the PandasDataFrameShim
defined in ingestion/ingest.py â€“ the pipeline uses whichever is available.

Transformations applied
-----------------------
  1. Drop rows where the target label is null.
  2. Impute remaining nulls: approximate median for each column.
  3. Feature engineering (3 new columns):
       age_group       : ordinal age bucket (0=Young, 1=Middle, 2=Senior, 3=Elder)
       chol_bp_ratio   : cholesterol / resting blood pressure
       heart_effort    : max_hr / (1 + ST_depression)
  4. Write cleaned data to Parquet (one file, easy to read back with pandas).

Run standalone:
    python -m transform.transform
"""

import os
import pandas as pd
import numpy as np

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(BASE_DIR, "data", "processed", "heart_clean.parquet")

NUMERIC_COLS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal",
]


# â”€â”€ core transform (operates on a pandas DataFrame) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _transform_pandas(pdf: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    All transformations applied to a plain pandas DataFrame.
    Called by transform() regardless of whether PySpark is available.
    """
    before = len(pdf)

    # 1. Drop rows with null target
    pdf = pdf.dropna(subset=["target"])
    print(f"[transform] Rows with null target dropped: {before - len(pdf)}")

    # 2. Impute nulls with column medians
    for col in pdf.columns:
        if pdf[col].isnull().any():
            median_val = pdf[col].median()
            pdf[col] = pdf[col].fillna(median_val)
            print(f"[transform]   Imputed '{col}' nulls with median={median_val:.3f}")

    # 3. Feature engineering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    # age_group: ordinal bucket (0=Young <40, 1=Middle <55, 2=Senior <65, 3=Elder)
    pdf["age_group"] = pd.cut(
        pdf["age"],
        bins=[-1, 39, 54, 64, 999],
        labels=[0, 1, 2, 3],
    ).astype(float)

    # chol_bp_ratio: cholesterol-to-blood-pressure ratio
    pdf["chol_bp_ratio"] = (pdf["chol"] / pdf["trestbps"]).astype(float)

    # heart_effort: max heart rate divided by (1 + ST depression)
    # Higher â†’ heart worked harder under stress relative to damage
    pdf["heart_effort"] = (pdf["thalach"] / (1.0 + pdf["oldpeak"])).astype(float)

    # 4. Cast all feature cols to float
    feature_cols = NUMERIC_COLS + ["age_group", "chol_bp_ratio", "heart_effort"]
    for col in feature_cols:
        pdf[col] = pdf[col].astype(float)

    # 5. Select final columns
    final_cols = feature_cols + ["target"]
    pdf = pdf[final_cols]

    print(f"[transform] Final shape: {pdf.shape[0]} rows Ã— {pdf.shape[1]} cols")
    print(pdf.dtypes.to_string())
    print()
    print(pdf.head(5).to_string())
    print()

    # 6. Write to Parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #  Spark writes a directory of part files; we do the same for compatibility
    parquet_dir = output_path
    os.makedirs(parquet_dir, exist_ok=True)
    out_file = os.path.join(parquet_dir, "part-00000.parquet")
    pdf.to_parquet(out_file, index=False)
    print(f"[transform] âœ“ Parquet written to: {out_file}")

    return pdf


# â”€â”€ public entry-point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def transform(df, output_path: str = PARQUET_PATH) -> pd.DataFrame:
    """
    Accept either:
      - a real PySpark DataFrame â†’ convert to pandas first, then transform
      - a PandasDataFrameShim   â†’ extract pandas df, then transform
      - a plain pandas DataFrame â†’ transform directly

    Returns the cleaned pandas DataFrame.
    """
    print("[transform] Starting transformation â€¦")

    # Normalise to pandas
    if hasattr(df, 'toPandas'):
        # Real PySpark DataFrame
        print("[transform] Converting Spark DataFrame to pandas â€¦")
        pdf = df.toPandas()
    elif hasattr(df, '_df'):
        # PandasDataFrameShim
        pdf = df._df.copy()
    elif isinstance(df, pd.DataFrame):
        pdf = df.copy()
    else:
        raise TypeError(f"Unsupported DataFrame type: {type(df)}")

    return _transform_pandas(pdf, output_path)


# â”€â”€ standalone execution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    from ingestion.ingest import get_spark, ingest

    spark  = get_spark()
    raw_df = ingest(spark)
    clean  = transform(raw_df)
    print("[transform] âœ“ Transform complete.")
    spark.stop()

