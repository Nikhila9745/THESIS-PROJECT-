# =============================================================================
# LINE-BY-LINE EXPLAINED VERSION of: explainability_5gb.py
# The code is unchanged. Every line or block has a comment above it saying what it does.
#
# WHAT THIS SCRIPT DOES IN ONE PARAGRAPH
# It answers "which input attributes does the trained Random Forest rely on most?" using
# PERMUTATION IMPORTANCE: take a balanced sample, score the model, then shuffle one column at a
# time (which breaks its link to the label) and see how much the F1 score drops. A big drop means
# the model depends heavily on that attribute. It saves a CSV and a bar chart of the top 15.
#
# NOTE: it needs the saved models ("models/preprocessing" and "models/RandomForest") produced by
# your training script (named in the header as stress_5GB_spark_ml.py, which is not one of the
# files you sent).
# =============================================================================

# explainability_5GB.py
# Permutation importance on the raw attributes for the 5GB run.
# Uses the saved preprocessing and Random Forest models in the results folder.
# Each attribute is shuffled on a balanced sample; the drop in F1 shows how much the model relies on it.
# Set CSV_PATH to the actual 5GB file. Needs the models saved by stress_5GB_spark_ml.py.
# Run from the Ubuntu terminal with the sparkenv active:  python3 explainability_5GB.py

# os: build file paths and create folders.
import os
# pandas: tables for the sample and the results.
import pandas as pd
# numpy: random shuffling and mean/standard deviation.
import numpy as np
# matplotlib: drawing the chart.
import matplotlib
# "Agg" = draw to files without needing a screen (works in a terminal/WSL).
matplotlib.use("Agg")
# pyplot is matplotlib's plotting interface.
import matplotlib.pyplot as plt
# Spark session and column functions.
from pyspark.sql import SparkSession, functions as F
# PipelineModel: a saved, already-fitted preprocessing pipeline that can be loaded.
from pyspark.ml import PipelineModel
# ImputerModel / StringIndexerModel: the fitted stage types, used to recover the column names.
from pyspark.ml.feature import ImputerModel, StringIndexerModel
# The type of the saved Random Forest.
from pyspark.ml.classification import RandomForestClassificationModel
# Computes accuracy / F1.
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ---------------- Settings ----------------
# Path to the 5 GB CSV.
CSV_PATH = "/mnt/d/THESIS PROJECT/Stress_Data_5GB.csv"
# Folder that contains the models saved by the training run.
RESULTS_DIR = "/mnt/c/Users/JP/THESIS PROJECT/5GB_Results"
# Column holding the class to predict.
TARGET = "Stress_Level"
# Class-name -> number mapping (the code below applies it with when()).
LABEL_MAP = {"Low": 0.0, "Medium": 1.0, "High": 2.0}
# Columns excluded in training. Defined here for reference; the actual feature lists come from the saved preprocessing.
EXCLUDE_COLS = ["Record_ID", "Stress_Score"]   # must match the training run
# Rows per class in the explanation sample.
SAMPLE_PER_CLASS = 2000
# How many times each attribute is shuffled (results are averaged to reduce randomness).
N_REPEATS = 3
# Random seed for repeatability.
SEED = 42

# Output folder for this script's results.
OUT_DIR = os.path.join(RESULTS_DIR, "explainability")
# Create it if needed.
os.makedirs(OUT_DIR, exist_ok=True)

# Start a local Spark session.
spark = (SparkSession.builder.appName("Explainability 5GB")
         # use all CPU cores,
         .master("local[*]")
         # 6 GB driver memory,
         .config("spark.driver.memory", "6g")
         # 16 shuffle partitions,
         .config("spark.sql.shuffle.partitions", "16")
         # create/reuse the session.
         .getOrCreate())
# Quieter logs.
spark.sparkContext.setLogLevel("WARN")

# try/finally so Spark is always stopped at the end.
try:
    # Load the models saved by the training run.
    # The fitted preprocessing pipeline (imputer, indexers, encoders, assembler, scaler).
    prep = PipelineModel.load(os.path.join(RESULTS_DIR, "models", "preprocessing"))
    # The fitted Random Forest.
    rf = RandomForestClassificationModel.load(os.path.join(RESULTS_DIR, "models", "RandomForest"))

    # The attribute lists come from the fitted preprocessing, so they always match the model.
    # Numeric attribute names = the input columns of the fitted Imputer stage.
    numeric_cols = list(next(s for s in prep.stages if isinstance(s, ImputerModel)).getInputCols())
    # Categorical attribute names = the input column of every fitted StringIndexer stage.
    cat_cols = [s.getInputCol() for s in prep.stages if isinstance(s, StringIndexerModel)]

    # Same cleaning as the training run.
    # Read the CSV with a header and inferred column types.
    raw = (spark.read.option("header", "true").option("inferSchema", "true").csv(CSV_PATH))
    # Trim spaces from the label text and keep only Low / Medium / High rows.
    df = raw.withColumn(TARGET, F.trim(F.col(TARGET))).filter(F.col(TARGET).isin(list(LABEL_MAP)))
    # Convert the label text to a number: Low=0, Medium=1, High=2.
    df = df.withColumn("label", F.when(F.col(TARGET) == "Low", 0.0)
                                 .when(F.col(TARGET) == "Medium", 1.0)
                                 .otherwise(2.0))
    # Cast the numeric attributes to double, as in training.
    for c in numeric_cols:
        df = df.withColumn(c, F.col(c).cast("double"))
    # Keep only the model attributes plus the label.
    df = df.select(*numeric_cols, *cat_cols, "label")

    # Balanced sample, small enough to permute in pandas.
    # Count rows per class.
    counts = {r["label"]: r["count"] for r in df.groupBy("label").count().collect()}
    # Fraction needed per class to reach about 2,000 rows (never above 1).
    fractions = {k: min(1.0, SAMPLE_PER_CLASS / v) for k, v in counts.items()}
    # Take the stratified sample.
    sample = df.sampleBy("label", fractions=fractions, seed=SEED)
    # Remember the Spark schema so shuffled copies can be turned back into Spark tables with identical types.
    schema = sample.schema
    # Bring the small sample into pandas.
    pdf = sample.toPandas()
    # Report the sample size.
    print(f"Explanation sample: {len(pdf)} records")

    # Evaluator for accuracy / F1 (metric chosen per call).
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    # Helper: score the model on a pandas table.
    def score(frame):
        # Convert the pandas table to a Spark table using the saved schema.
        sdf = spark.createDataFrame(frame, schema=schema)
        # Preprocess, then predict with the Random Forest.
        predicted = rf.transform(prep.transform(sdf))
        # Return accuracy and weighted F1.
        return {
            "accuracy": evaluator.evaluate(predicted, {evaluator.metricName: "accuracy"}),
            "weighted_f1": evaluator.evaluate(predicted, {evaluator.metricName: "f1"}),
        }

    # Score on the untouched sample: the reference ("baseline") performance.
    baseline = score(pdf)
    # Print the baseline.
    print("Baseline:", baseline)

    # Random number generator with a fixed seed for the shuffles.
    rng = np.random.default_rng(SEED)
    # Will hold one result row per attribute.
    rows = []
    # All attributes to test: numeric first, then categorical.
    attributes = numeric_cols + cat_cols
    # Loop over attributes; i counts from 1 for the progress message.
    for i, col in enumerate(attributes, 1):
        # Lists of score drops across the repeats.
        drops_f1, drops_acc = [], []
        # Repeat the shuffle several times.
        for _ in range(N_REPEATS):
            # Fresh copy so the original sample is never changed.
            shuffled = pdf.copy()
            # Randomly reorder this one column's values across rows (its link to the label is broken).
            shuffled[col] = rng.permutation(shuffled[col].values)
            # Score the model on the shuffled data.
            s = score(shuffled)
            # How much weighted F1 dropped versus the baseline.
            drops_f1.append(baseline["weighted_f1"] - s["weighted_f1"])
            # How much accuracy dropped versus the baseline.
            drops_acc.append(baseline["accuracy"] - s["accuracy"])
        # Label the attribute type for the results table.
        kind = "numeric" if col in numeric_cols else "categorical"
        # Store mean and spread of the drops for this attribute.
        rows.append({"attribute": col, "type": kind,
                     "f1_drop_mean": float(np.mean(drops_f1)),
                     "f1_drop_sd": float(np.std(drops_f1)),
                     "accuracy_drop_mean": float(np.mean(drops_acc))})
        # Progress line, e.g. [3/36] Sleep_Hours: F1 drop 0.0123
        print(f"[{i}/{len(attributes)}] {col}: F1 drop {np.mean(drops_f1):.4f}")

    # Build the results table, biggest F1 drop (most important attribute) first.
    result = pd.DataFrame(rows).sort_values("f1_drop_mean", ascending=False)
    # Save the full ranking.
    result.to_csv(os.path.join(OUT_DIR, "permutation_importance_raw.csv"), index=False)

    # Take the 15 most important attributes and reverse the order so the biggest bar is on top of the chart.
    top = result.head(15).iloc[::-1]
    # Create a figure 7 x 5.5 inches.
    fig, ax = plt.subplots(figsize=(7, 5.5))
    # Horizontal bars = mean F1 drop; error bars = standard deviation across repeats.
    ax.barh(top["attribute"], top["f1_drop_mean"], xerr=top["f1_drop_sd"], color="#3B6EA8")
    # Axis label.
    ax.set_xlabel("Drop in weighted F1 when the attribute is shuffled")
    # Chart title.
    ax.set_title("5GB run: permutation importance (raw attributes)")
    # Light vertical grid lines for easier reading.
    ax.grid(axis="x", alpha=0.3)
    # Fit everything inside the figure.
    fig.tight_layout()
    # Save the chart as a 200-dpi PNG.
    fig.savefig(os.path.join(OUT_DIR, "permutation_importance_top15.png"), dpi=200)

    # Print where the files were saved and the top 15 table.
    print("\nSaved to:", OUT_DIR)
    print(result.head(15).to_string(index=False))
# Always stop Spark.
finally:
    spark.stop()
