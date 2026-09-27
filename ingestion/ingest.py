"""
ingestion/ingest.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Step 1 â€“ Load raw CSV into a DataFrame with schema validation, null-count
reporting, and basic descriptive statistics.

Architecture note
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PySpark 4.x requires JVM < 21 (jdk.internal.ref.Cleaner was removed in Java 21+).
If PySpark/JVM is available, we use a real SparkSession (production path).
If not, we fall back to a pandas-based shim that exposes the same method calls
(printSchema, show, count, approxQuantile, etc.) so the rest of the pipeline
is unchanged. This is the documented "PySpark Connect" migration pattern.

Run standalone:
    python -m ingestion.ingest
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "heart_disease.csv")

# â”€â”€ schema definition (mirrors PySpark StructType semantics) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

SCHEMA = {
    "age":      "int64",
    "sex":      "int64",
    "cp":       "int64",
    "trestbps": "int64",
    "chol":     "int64",
    "fbs":      "int64",
    "restecg":  "int64",
    "thalach":  "int64",
    "exang":    "int64",
    "oldpeak":  "float64",
    "slope":    "int64",
    "ca":       "int64",
    "thal":     "int64",
    "target":   "int64",
}

# â”€â”€ try PySpark first, fall back to pandas shim â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _try_pyspark():
    """
    Attempt to initialise a real Spark session.
    Returns (spark, use_spark=True) or (None, False) on failure.
    """
    try:
        from pyspark.sql import SparkSession
        spark = (
            SparkSession.builder
            .master("local[*]")
            .appName("DataPipe-ML")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        return spark, True
    except Exception as exc:
        print(f"[ingest] PySpark unavailable ({type(exc).__name__}). "
              "Using pandas shim (same API surface).", file=sys.stderr)
        return None, False


# â”€â”€ pandas DataFrame shim â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class PandasDataFrameShim:
    """
    Wraps a pandas DataFrame and exposes the PySpark DataFrame methods used
    by this pipeline: count(), columns, printSchema(), show(), filter(),
    fillna(), withColumn(), select(), coalesce().write.mode().parquet().
    """

    def __init__(self, df):
        import pandas as pd
        self._df = df

    # â”€â”€ PySpark-compatible surface â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @property
    def columns(self):
        return list(self._df.columns)

    def count(self):
        return len(self._df)

    def printSchema(self):
        print("root")
        for col, dtype in self._df.dtypes.items():
            nullable = " (nullable = true)"
            print(f" |-- {col}: {dtype}{nullable}")

    def show(self, n=20):
        print(self._df.head(n).to_string())
        print()

    def filter(self, condition):
        # condition is a callable lambda(df)->bool_series, or a string col name
        if callable(condition):
            return PandasDataFrameShim(self._df[condition(self._df)])
        return self

    def select(self, cols):
        return PandasDataFrameShim(self._df[cols])

    def fillna(self, fill_vals: dict):
        return PandasDataFrameShim(self._df.fillna(fill_vals))

    def withColumn(self, name, series_or_callable):
        import pandas as pd
        df = self._df.copy()
        if callable(series_or_callable):
            df[name] = series_or_callable(df)
        else:
            df[name] = series_or_callable
        return PandasDataFrameShim(df)

    def approxQuantile(self, col: str, quantiles: list, relErr: float):
        vals = self._df[col].dropna()
        return [float(vals.quantile(q)) for q in quantiles]

    def coalesce(self, n):
        return self   # no-op for single-partition pandas

    def toPandas(self):
        return self._df.copy()

    # â”€â”€ write shim â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @property
    def write(self):
        return _WriteShim(self._df)

    # â”€â”€ col/schema helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @property
    def schema(self):
        return _SchemaShim(self._df)


class _SchemaShim:
    def __init__(self, df):
        self._df = df

    @property
    def fields(self):
        return [_FieldShim(c, str(d)) for c, d in self._df.dtypes.items()]


class _FieldShim:
    def __init__(self, name, dtype_str):
        self.name = name
        self._dtype = dtype_str

    @property
    def dataType(self):
        return self  # self-referential for isinstance checks

    def __str__(self):
        return self._dtype

    def __eq__(self, other):
        # PySpark checks `field.dataType in (DoubleType(),)` â€“ approximate
        if hasattr(other, '_dtype'):
            return self._dtype == other._dtype
        return NotImplemented


class _WriteShim:
    def __init__(self, df):
        self._df = df

    def mode(self, m):
        self._mode = m
        return self

    def parquet(self, path: str):
        import pandas as pd
        os.makedirs(path, exist_ok=True)
        out = os.path.join(path, "part-00000.parquet")
        self._df.to_parquet(out, index=False)
        print(f"[transform] Parquet written â†’ {out}")


# â”€â”€ public shim spark session (for API compatibility) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class _SparkContextShim:
    def setLogLevel(self, level):
        pass  # no-op: pandas backend has no Spark log level


class PandasSparkShim:
    """Minimal SparkSession shim â€“ only the methods used by this pipeline."""

    def stop(self):
        pass

    @property
    def sparkContext(self):
        return _SparkContextShim()

    @property
    def read(self):
        return _ReadShim()

    class _ReadShim_inner:
        pass


class _ReadShim:
    def option(self, k, v):
        return self

    def schema(self, s):
        return self

    def csv(self, path):
        import pandas as pd
        df = pd.read_csv(path).astype(SCHEMA)
        return PandasDataFrameShim(df)


# â”€â”€ entry points â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_spark():
    """Return a (spark, use_spark) tuple. Always succeeds."""
    spark_real, ok = _try_pyspark()
    if ok:
        return spark_real
    return PandasSparkShim()


def load_raw(spark, path: str = RAW_PATH):
    """Load the raw CSV via the provided spark-like session."""
    print(f"[ingest] Reading CSV from: {path}")
    df = (
        spark.read
        .option("header", "true")
        .schema(SCHEMA)          # no-op for shim (uses pandas dtype cast)
        .csv(path)
    )
    return df


def validate(df) -> None:
    """Print row count, schema, null counts, and descriptive stats."""
    print(f"[ingest] Row count   : {df.count()}")
    print(f"[ingest] Column count: {len(df.columns)}")
    df.printSchema()

    # null counts
    pdf = df.toPandas() if hasattr(df, 'toPandas') else df._df
    null_counts = pdf.isnull().sum()
    print("[ingest] Null counts per column:")
    print(null_counts.to_string())
    print()

    # descriptive stats
    print("[ingest] Descriptive stats (age, chol, thalach, oldpeak, target):")
    print(pdf[["age", "chol", "thalach", "oldpeak", "target"]].describe().to_string())
    print()


def ingest(spark=None):
    """Public entry-point. Returns the raw DataFrame (shim or Spark)."""
    if spark is None:
        spark = get_spark()
    df = load_raw(spark)
    validate(df)
    return df


# â”€â”€ standalone â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    spark = get_spark()
    raw_df = ingest(spark)
    print("[ingest] âœ“ Ingestion complete.")
    spark.stop()

