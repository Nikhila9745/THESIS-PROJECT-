# =============================================================================
# LINE-BY-LINE EXPLAINED VERSION of: early_prediction_5gb.py
# The code is unchanged. Every line or block has a comment above it saying what it does.
#
# WHAT THIS SCRIPT DOES IN ONE PARAGRAPH
# It sorts the 5 GB stress dataset by time, uses the earliest 80% of records to train two
# models (Logistic Regression and Random Forest), and tests them on the latest 20%. This is a
# "temporal holdout": it checks whether a pattern learned from older records still works on
# newer ones. It is NOT early prediction for a person, because the dataset has no participant ID.
# =============================================================================

# early_prediction_5GB.py
# Temporal holdout test: train on the earlier records, test on the later records.
#
# What this tests: whether the pattern learned from earlier records still holds for later records.
# What it does NOT test: whether stress can be predicted before it is recorded for the same person.
# That needs a participant ID and time-stamped measurements taken before each stress report.
# The dataset has no participant ID, so the results below must not be described as early prediction.
#
# Set CSV_PATH to the actual 5GB file. Results go to the 5GB results folder.
# Run from the Ubuntu terminal with the sparkenv active:  python3 early_prediction_5GB.py

# os: create folders and build file paths.
import os
# pandas: used for the final small results table and for converting the cutoff number into a readable date.
import pandas as pd
# SparkSession starts Spark; functions (F) gives column operations such as col(), when(), to_timestamp().
from pyspark.sql import SparkSession, functions as F
# NumericType lets us detect which columns are numbers (int, double, ...).
from pyspark.sql.types import NumericType
# Pipeline chains preprocessing steps so they run in order and can be fitted once.
from pyspark.ml import Pipeline
# The preprocessing building blocks:
#   Imputer = fill missing numbers; StringIndexer = text category -> number;
#   OneHotEncoder = number -> 0/1 columns; VectorAssembler = join columns into one vector;
#   StandardScaler = rescale features.
from pyspark.ml.feature import (Imputer, StringIndexer, OneHotEncoder,
                                VectorAssembler, StandardScaler)
# The two classifiers that will be compared.
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
# Computes accuracy and F1 for multi-class predictions.
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ---------------- Settings ----------------
# Path to the 5 GB CSV (WSL path: /mnt/d = the D: drive in Windows).
CSV_PATH = "/mnt/d/THESIS PROJECT/Stress_Data_5GB.csv"
# Folder where results are saved (/mnt/c = the C: drive in Windows).
OUT_DIR = "/mnt/c/Users/JP/THESIS PROJECT/5GB_Results/early_prediction"
# Name of the column holding the class we want to predict (Low / Medium / High).
TARGET = "Stress_Level"
# Name of the column holding the time of each record.
TIME_COL = "Timestamp"
# Mapping from class names to numbers (the code below applies the same mapping with when()).
LABEL_MAP = {"Low": 0.0, "Medium": 1.0, "High": 2.0}
# Columns that must NOT be used as features: an ID, the raw score (the label is derived from it = leakage), and the time column.
EXCLUDE_COLS = ["Record_ID", "Stress_Score", TIME_COL]   # Stress_Score is the score the label is derived from
# A text column is only treated as a category if it has at most 50 distinct values.
MAX_CAT_LEVELS = 50
# The 80th percentile of time is the split point between "earlier" (train) and "later" (test) records.
CUTOFF_QUANTILE = 0.8       # records before the 80th percentile of time are training data
# Up to 10,000 training examples per class.
PER_CLASS_TRAIN = 10000
# Up to 2,000 test examples per class.
PER_CLASS_TEST = 2000
# At least 95% of timestamps must be readable, otherwise the script stops.
MIN_PARSED_SHARE = 0.95
# Random seed so sampling and the forest are repeatable.
SEED = 42

# Create the output folder if it does not exist yet.
os.makedirs(OUT_DIR, exist_ok=True)
# Start Spark:
spark = (SparkSession.builder.appName("Early prediction 5GB")
         # run locally using all CPU cores,
         .master("local[*]")
         # give the driver 6 GB of memory,
         .config("spark.driver.memory", "6g")
         # use 16 partitions after shuffles (small, suits a laptop),
         .config("spark.sql.shuffle.partitions", "16")
         # and create the session (or reuse an existing one).
         .getOrCreate())
# Only show warnings and errors in the console.
spark.sparkContext.setLogLevel("WARN")

# try/finally guarantees spark.stop() runs at the end even if something fails.
try:
    # Read the CSV: first row is the header; inferSchema makes Spark guess column types (numbers vs text).
    raw = spark.read.option("header", "true").option("inferSchema", "true").csv(CSV_PATH)

    # Parse the time column, trying the common formats.
    # coalesce picks, for each row, the first format that gives a valid timestamp.
    ts = F.coalesce(F.to_timestamp(F.col(TIME_COL), "yyyy-MM-dd HH:mm:ss"),
                    F.to_timestamp(F.col(TIME_COL), "yyyy-MM-dd'T'HH:mm:ss"),
                    F.to_timestamp(F.col(TIME_COL), "dd/MM/yyyy HH:mm"),
                    F.to_timestamp(F.col(TIME_COL), "yyyy-MM-dd"))
    # Add the parsed time as "_ts" and trim stray spaces from the stress label text.
    df = raw.withColumn("_ts", ts).withColumn(TARGET, F.trim(F.col(TARGET)))
    # Keep only rows whose label is exactly Low, Medium or High (drops blanks / typos).
    df = df.filter(F.col(TARGET).isin(list(LABEL_MAP)))
    # Number of valid-label rows.
    total = df.count()
    # How many of those have a readable timestamp.
    parsed = df.filter(F.col("_ts").isNotNull()).count()
    # Fraction parsed (0 if there are no rows, to avoid dividing by zero).
    share = parsed / total if total else 0
    # Report the parse rate.
    print(f"Timestamps parsed: {parsed:,} of {total:,} ({share:.1%})")
    # Safety check: stop if too many timestamps failed to parse.
    if share < MIN_PARSED_SHARE:
        raise SystemExit("Too few timestamps could be parsed. Check the format of the Timestamp column.")

    # Keep only rows with a valid time, and add "_ts_num" = the time as a number of seconds (easy to compare).
    df = df.filter(F.col("_ts").isNotNull()).withColumn("_ts_num", F.unix_timestamp("_ts"))
    # Find the time value below which 80% of records lie (approximate, error 0.1%).
    cutoff = df.approxQuantile("_ts_num", [CUTOFF_QUANTILE], 0.001)[0]
    # Convert that number back into a readable date for printing and saving.
    cutoff_value = pd.to_datetime(cutoff, unit="s")
    # Report the split point.
    print("Training records: before", cutoff_value, "| test records: from", cutoff_value)

    # Create the numeric label: Low=0, Medium=1, anything else (High)=2.
    df = df.withColumn("label", F.when(F.col(TARGET) == "Low", 0.0)
                                 .when(F.col(TARGET) == "Medium", 1.0)
                                 .otherwise(2.0))
    # Feature columns = every column except the label, helper time columns, the text target, and the excluded columns.
    features = [c for c in df.columns if c not in ("label", "_ts", "_ts_num", TARGET, *EXCLUDE_COLS)]
    # Among the features, the ones whose type is numeric.
    numeric_cols = [f.name for f in df.schema.fields
                    if f.name in features and isinstance(f.dataType, NumericType)]
    # The remaining features are text columns.
    string_cols = [c for c in features if c not in numeric_cols]
    # Count approximately how many distinct values each text column has (skipped if there are no text columns).
    distinct = df.agg(*[F.approx_count_distinct(c).alias(c) for c in string_cols]).first().asDict() if string_cols else {}
    # Keep only text columns with <= 50 distinct values as categories. Free-text columns (like a comment) are silently left out.
    cat_cols = [c for c in string_cols if distinct[c] <= MAX_CAT_LEVELS]
    # Cast every numeric column to double so Spark treats them uniformly.
    for c in numeric_cols:
        df = df.withColumn(c, F.col(c).cast("double"))

    # Earlier records (strictly before the cutoff) -> training side.
    train_side = df.filter(F.col("_ts_num") < cutoff)
    # Later records (at or after the cutoff) -> test side.
    test_side = df.filter(F.col("_ts_num") >= cutoff)

    # Helper that returns a class-balanced sample of up to per_class rows per label.
    def balance(frame, per_class):
        # Count rows per label.
        counts = {r["label"]: r["count"] for r in frame.groupBy("label").count().collect()}
        # Fraction to keep per label: per_class / available, never more than 1.
        fractions = {k: min(1.0, per_class / v) for k, v in counts.items()}
        # Stratified random sample (the sizes are approximate, not exact).
        return frame.sampleBy("label", fractions=fractions, seed=SEED)

    # Balanced training sample, using only the model columns + label; cache() keeps it in memory for reuse.
    train_raw = balance(train_side.select(*numeric_cols, *cat_cols, "label"), PER_CLASS_TRAIN).cache()
    # Balanced test sample, built the same way from the later records.
    test_raw = balance(test_side.select(*numeric_cols, *cat_cols, "label"), PER_CLASS_TEST).cache()
    # Report how many rows ended up in each sample.
    print("Balanced training records:", train_raw.count(), "| balanced test records:", test_raw.count())

    # Preprocessing is fitted on the earlier records only.
    # Stage 1: fill missing numbers (Imputer's default strategy is the mean); outputs get an "_imp" suffix.
    stages = [Imputer(inputCols=numeric_cols, outputCols=[f"{c}_imp" for c in numeric_cols])]
    # For each category column add a StringIndexer (text -> number); "keep" gives unseen values their own bucket instead of crashing.
    for c in cat_cols:
        stages.append(StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep"))
    # If there are category columns, one-hot encode all the indexed columns at once.
    if cat_cols:
        stages.append(OneHotEncoder(inputCols=[f"{c}_idx" for c in cat_cols],
                                    outputCols=[f"{c}_ohe" for c in cat_cols],
                                    handleInvalid="keep"))
    # Names of all columns that go into the feature vector: imputed numbers + one-hot columns.
    assembled = [f"{c}_imp" for c in numeric_cols] + [f"{c}_ohe" for c in cat_cols]
    # Combine them into one vector column; "skip" drops any row that still has an invalid value.
    stages.append(VectorAssembler(inputCols=assembled, outputCol="features_raw", handleInvalid="skip"))
    # Divide each feature by its standard deviation (no centering, which keeps one-hot vectors sparse).
    stages.append(StandardScaler(inputCol="features_raw", outputCol="features", withMean=False, withStd=True))
    # Fit the whole preprocessing pipeline on the TRAINING sample only (so nothing is learned from the test period).
    prep = Pipeline(stages=stages).fit(train_raw)

    # Apply the fitted preprocessing to the training sample and keep only the two columns the models need.
    train = prep.transform(train_raw).select("features", "label")
    # Apply the SAME fitted preprocessing to the test sample.
    test = prep.transform(test_raw).select("features", "label")

    # Evaluator used to compute accuracy and F1 (the metric is chosen per call below).
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    # The two models to compare.
    models = {
        # Multi-class logistic regression, 30 iterations, regularisation strength 0.05.
        "LogisticRegression": LogisticRegression(maxIter=30, regParam=0.05, family="multinomial"),
        # Random forest: 30 trees, depth 8, each tree sees 70% of rows and sqrt(#features) features per split.
        "RandomForest": RandomForestClassifier(numTrees=30, maxDepth=8, subsamplingRate=0.7,
                                               featureSubsetStrategy="sqrt", seed=SEED),
    }
    # List that will hold one result row per model.
    rows = []
    # Train and evaluate each model in turn.
    for name, est in models.items():
        # Train on the earlier records.
        model = est.fit(train)
        # Predict on the later records.
        pred = model.transform(test)
        # Record the scores and basic info.
        rows.append({
            "model": name,
            # Fraction of correct predictions.
            "accuracy": evaluator.evaluate(pred, {evaluator.metricName: "accuracy"}),
            # F1 averaged across the classes, weighted by class size.
            "weighted_f1": evaluator.evaluate(pred, {evaluator.metricName: "f1"}),
            # The date used to split train and test.
            "train_cutoff": str(cutoff_value),
            # Size of the training set actually used.
            "train_records": train.count(),
            # Size of the test set actually used.
            "test_records": test.count(),
        })
        # Print the result for this model.
        print(rows[-1])

    # Turn the list of results into a small table.
    result = pd.DataFrame(rows)
    # Save the table as a CSV.
    result.to_csv(os.path.join(OUT_DIR, "temporal_holdout_results.csv"), index=False)
    # Also write a short README next to the results so nobody mistakes this for early prediction.
    with open(os.path.join(OUT_DIR, "README.txt"), "w", encoding="utf-8") as f:
        f.write("Temporal holdout: trained on records before the cutoff, tested on later records.\n"
                "This tests whether the learned pattern holds over time. It is not early prediction\n"
                "for the same person, because the dataset has no participant identifier and the\n"
                "features and stress label are recorded together.\n")
    # Print where the files went, then the results table.
    print("\nSaved to:", OUT_DIR)
    print(result.to_string(index=False))
# Always shut Spark down, whether the run succeeded or failed.
finally:
    spark.stop()
