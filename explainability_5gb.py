# explainability_5GB.py
# Permutation importance on the raw attributes for the 5GB run.
# Uses the saved preprocessing and Random Forest models in the results folder.
# Each attribute is shuffled on a balanced sample; the drop in F1 shows how much the model relies on it.
# Set CSV_PATH to the actual 5GB file. Needs the models saved by stress_5GB_spark_ml.py.
# Run from the Ubuntu terminal with the sparkenv active:  python3 explainability_5GB.py

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.feature import ImputerModel, StringIndexerModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ---------------- Settings ----------------
CSV_PATH = "/mnt/d/THESIS PROJECT/Stress_Data_5GB.csv"
RESULTS_DIR = "/mnt/c/Users/JP/THESIS PROJECT/5GB_Results"
TARGET = "Stress_Level"
LABEL_MAP = {"Low": 0.0, "Medium": 1.0, "High": 2.0}
EXCLUDE_COLS = ["Record_ID", "Stress_Score"]   # must match the training run
SAMPLE_PER_CLASS = 2000
N_REPEATS = 3
SEED = 42

OUT_DIR = os.path.join(RESULTS_DIR, "explainability")
os.makedirs(OUT_DIR, exist_ok=True)

spark = (SparkSession.builder.appName("Explainability 5GB")
         .master("local[*]")
         .config("spark.driver.memory", "6g")
         .config("spark.sql.shuffle.partitions", "16")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

try:
    # Load the models saved by the training run.
    prep = PipelineModel.load(os.path.join(RESULTS_DIR, "models", "preprocessing"))
    rf = RandomForestClassificationModel.load(os.path.join(RESULTS_DIR, "models", "RandomForest"))

    # The attribute lists come from the fitted preprocessing, so they always match the model.
    numeric_cols = list(next(s for s in prep.stages if isinstance(s, ImputerModel)).getInputCols())
    cat_cols = [s.getInputCol() for s in prep.stages if isinstance(s, StringIndexerModel)]

    # Same cleaning as the training run.
    raw = (spark.read.option("header", "true").option("inferSchema", "true").csv(CSV_PATH))
    df = raw.withColumn(TARGET, F.trim(F.col(TARGET))).filter(F.col(TARGET).isin(list(LABEL_MAP)))
    df = df.withColumn("label", F.when(F.col(TARGET) == "Low", 0.0)
                                 .when(F.col(TARGET) == "Medium", 1.0)
                                 .otherwise(2.0))
    for c in numeric_cols:
        df = df.withColumn(c, F.col(c).cast("double"))
    df = df.select(*numeric_cols, *cat_cols, "label")

    # Balanced sample, small enough to permute in pandas.
    counts = {r["label"]: r["count"] for r in df.groupBy("label").count().collect()}
    fractions = {k: min(1.0, SAMPLE_PER_CLASS / v) for k, v in counts.items()}
    sample = df.sampleBy("label", fractions=fractions, seed=SEED)
    schema = sample.schema
    pdf = sample.toPandas()
    print(f"Explanation sample: {len(pdf)} records")

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    def score(frame):
        sdf = spark.createDataFrame(frame, schema=schema)
        predicted = rf.transform(prep.transform(sdf))
        return {
            "accuracy": evaluator.evaluate(predicted, {evaluator.metricName: "accuracy"}),
            "weighted_f1": evaluator.evaluate(predicted, {evaluator.metricName: "f1"}),
        }

    baseline = score(pdf)
    print("Baseline:", baseline)

    rng = np.random.default_rng(SEED)
    rows = []
    attributes = numeric_cols + cat_cols
    for i, col in enumerate(attributes, 1):
        drops_f1, drops_acc = [], []
        for _ in range(N_REPEATS):
            shuffled = pdf.copy()
            shuffled[col] = rng.permutation(shuffled[col].values)
            s = score(shuffled)
            drops_f1.append(baseline["weighted_f1"] - s["weighted_f1"])
            drops_acc.append(baseline["accuracy"] - s["accuracy"])
        kind = "numeric" if col in numeric_cols else "categorical"
        rows.append({"attribute": col, "type": kind,
                     "f1_drop_mean": float(np.mean(drops_f1)),
                     "f1_drop_sd": float(np.std(drops_f1)),
                     "accuracy_drop_mean": float(np.mean(drops_acc))})
        print(f"[{i}/{len(attributes)}] {col}: F1 drop {np.mean(drops_f1):.4f}")

    result = pd.DataFrame(rows).sort_values("f1_drop_mean", ascending=False)
    result.to_csv(os.path.join(OUT_DIR, "permutation_importance_raw.csv"), index=False)

    top = result.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(top["attribute"], top["f1_drop_mean"], xerr=top["f1_drop_sd"], color="#3B6EA8")
    ax.set_xlabel("Drop in weighted F1 when the attribute is shuffled")
    ax.set_title("5GB run: permutation importance (raw attributes)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "permutation_importance_top15.png"), dpi=200)

    print("\nSaved to:", OUT_DIR)
    print(result.head(15).to_string(index=False))
finally:
    spark.stop()
