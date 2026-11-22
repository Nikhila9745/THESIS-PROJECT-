# early_prediction_20gb.py
# Temporal holdout test: train on the earlier records, test on the later records.
#
# What this tests: whether the pattern learned from earlier records still holds for later records.
# What it does NOT test: whether stress can be predicted before it is recorded for the same person.
# That needs a participant ID and time-stamped measurements taken before each stress report.
# The dataset has no participant ID, so the results below must not be described as early prediction.
#
# Set CSV_PATH to the actual 20GB file. Results go to the 20GB results folder.
# Run from the Ubuntu terminal with the sparkenv active:  python3 early_prediction_20gb.py

import os
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import NumericType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (Imputer, StringIndexer, OneHotEncoder,
                                VectorAssembler, StandardScaler)
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ---------------- Settings ----------------
CSV_PATH = "/mnt/c/Users/JP/THESIS PROJECT/Stress_Data_20GB.csv"
OUT_DIR = "/mnt/c/Users/JP/THESIS PROJECT/20GB_Results/early_prediction"
TARGET = "Stress_Level"
TIME_COL = "Timestamp"
LABEL_MAP = {"Low": 0.0, "Medium": 1.0, "High": 2.0}
EXCLUDE_COLS = ["Record_ID", "Stress_Score", TIME_COL]   # Stress_Score is the score the label is derived from
MAX_CAT_LEVELS = 50
CUTOFF_QUANTILE = 0.8       # records before the 80th percentile of time are training data
PER_CLASS_TRAIN = 10000
PER_CLASS_TEST = 2000
MIN_PARSED_SHARE = 0.95
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)
spark = (SparkSession.builder.appName("Early prediction 20GB")
         .master("local[*]")
         .config("spark.driver.memory", "6g")
         .config("spark.sql.shuffle.partitions", "16")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

try:
    raw = spark.read.option("header", "true").option("inferSchema", "true").csv(CSV_PATH)

    # Parse the time column, trying the common formats.
    ts = F.coalesce(F.to_timestamp(F.col(TIME_COL), "yyyy-MM-dd HH:mm:ss"),
                    F.to_timestamp(F.col(TIME_COL), "yyyy-MM-dd'T'HH:mm:ss"),
                    F.to_timestamp(F.col(TIME_COL), "dd/MM/yyyy HH:mm"),
                    F.to_timestamp(F.col(TIME_COL), "yyyy-MM-dd"))
    df = raw.withColumn("_ts", ts).withColumn(TARGET, F.trim(F.col(TARGET)))
    df = df.filter(F.col(TARGET).isin(list(LABEL_MAP)))
    total = df.count()
    parsed = df.filter(F.col("_ts").isNotNull()).count()
    share = parsed / total if total else 0
    print(f"Timestamps parsed: {parsed:,} of {total:,} ({share:.1%})")
    if share < MIN_PARSED_SHARE:
        raise SystemExit("Too few timestamps could be parsed. Check the format of the Timestamp column.")

    df = df.filter(F.col("_ts").isNotNull()).withColumn("_ts_num", F.unix_timestamp("_ts"))
    cutoff = df.approxQuantile("_ts_num", [CUTOFF_QUANTILE], 0.001)[0]
    cutoff_value = pd.to_datetime(cutoff, unit="s")
    print("Training records: before", cutoff_value, "| test records: from", cutoff_value)

    df = df.withColumn("label", F.when(F.col(TARGET) == "Low", 0.0)
                                 .when(F.col(TARGET) == "Medium", 1.0)
                                 .otherwise(2.0))
    features = [c for c in df.columns if c not in ("label", "_ts", "_ts_num", TARGET, *EXCLUDE_COLS)]
    numeric_cols = [f.name for f in df.schema.fields
                    if f.name in features and isinstance(f.dataType, NumericType)]
    string_cols = [c for c in features if c not in numeric_cols]
    distinct = df.agg(*[F.approx_count_distinct(c).alias(c) for c in string_cols]).first().asDict() if string_cols else {}
    cat_cols = [c for c in string_cols if distinct[c] <= MAX_CAT_LEVELS]
    for c in numeric_cols:
        df = df.withColumn(c, F.col(c).cast("double"))

    train_side = df.filter(F.col("_ts_num") < cutoff)
    test_side = df.filter(F.col("_ts_num") >= cutoff)

    def balance(frame, per_class):
        counts = {r["label"]: r["count"] for r in frame.groupBy("label").count().collect()}
        fractions = {k: min(1.0, per_class / v) for k, v in counts.items()}
        return frame.sampleBy("label", fractions=fractions, seed=SEED)

    train_raw = balance(train_side.select(*numeric_cols, *cat_cols, "label"), PER_CLASS_TRAIN).cache()
    test_raw = balance(test_side.select(*numeric_cols, *cat_cols, "label"), PER_CLASS_TEST).cache()
    print("Balanced training records:", train_raw.count(), "| balanced test records:", test_raw.count())

    # Preprocessing is fitted on the earlier records only.
    stages = [Imputer(inputCols=numeric_cols, outputCols=[f"{c}_imp" for c in numeric_cols])]
    for c in cat_cols:
        stages.append(StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep"))
    if cat_cols:
        stages.append(OneHotEncoder(inputCols=[f"{c}_idx" for c in cat_cols],
                                    outputCols=[f"{c}_ohe" for c in cat_cols],
                                    handleInvalid="keep"))
    assembled = [f"{c}_imp" for c in numeric_cols] + [f"{c}_ohe" for c in cat_cols]
    stages.append(VectorAssembler(inputCols=assembled, outputCol="features_raw", handleInvalid="skip"))
    stages.append(StandardScaler(inputCol="features_raw", outputCol="features", withMean=False, withStd=True))
    prep = Pipeline(stages=stages).fit(train_raw)

    train = prep.transform(train_raw).select("features", "label")
    test = prep.transform(test_raw).select("features", "label")

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    models = {
        "LogisticRegression": LogisticRegression(maxIter=30, regParam=0.05, family="multinomial"),
        "RandomForest": RandomForestClassifier(numTrees=30, maxDepth=8, subsamplingRate=0.7,
                                               featureSubsetStrategy="sqrt", seed=SEED),
    }
    rows = []
    for name, est in models.items():
        model = est.fit(train)
        pred = model.transform(test)
        rows.append({
            "model": name,
            "accuracy": evaluator.evaluate(pred, {evaluator.metricName: "accuracy"}),
            "weighted_f1": evaluator.evaluate(pred, {evaluator.metricName: "f1"}),
            "train_cutoff": str(cutoff_value),
            "train_records": train.count(),
            "test_records": test.count(),
        })
        print(rows[-1])

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(OUT_DIR, "temporal_holdout_results.csv"), index=False)
    with open(os.path.join(OUT_DIR, "README.txt"), "w", encoding="utf-8") as f:
        f.write("Temporal holdout: trained on records before the cutoff, tested on later records.\n"
                "This tests whether the learned pattern holds over time. It is not early prediction\n"
                "for the same person, because the dataset has no participant identifier and the\n"
                "features and stress label are recorded together.\n")
    print("\nSaved to:", OUT_DIR)
    print(result.to_string(index=False))
finally:
    spark.stop()
