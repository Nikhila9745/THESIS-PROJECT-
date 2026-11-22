# stress_5GB_spark_ml.py
# Plain PySpark + Spark ML pipeline for the 5GB student stress CSV.
# Run from the Ubuntu terminal with the sparkenv active:  python3 stress_5GB_spark_ml.py
# Needs ~12 GB RAM for WSL (see .wslconfig). Duplicates are removed after balancing, before the split.

import os
import json
import time
from datetime import datetime

import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import NumericType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (Imputer, StringIndexer, OneHotEncoder,
                                OneHotEncoderModel, VectorAssembler, StandardScaler)
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ---------------- Settings ----------------
CSV_PATH = "/mnt/d/THESIS PROJECT/Stress_Data_5GB.csv"  
OUTPUT_DIR = "/mnt/c/Users/JP/THESIS PROJECT/5GB_Results"
TARGET = "Stress_Level"
LABEL_MAP = {"Low": 0.0, "Medium": 1.0, "High": 2.0}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}
PER_CLASS = 10000          # balanced sample size per class
MAX_CAT_LEVELS = 50        # text columns with more distinct values are dropped
SEED = 42
# Columns that must not be used as features: IDs, and the score the label is derived from.
EXCLUDE_COLS = ["Record_ID", "Stress_Score"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
t_start = time.time()
run = {"csv": CSV_PATH, "started": datetime.now().isoformat(timespec="seconds")}

spark = (SparkSession.builder
         .appName("Student Stress 5GB")
         .master("local[*]")
         .config("spark.driver.memory", "6g")
         .config("spark.sql.shuffle.partitions", "16")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

try:
    # ---------- Step 1: Load ----------
    print("Step 1: loading CSV")
    raw = (spark.read.option("header", "true")
           .option("inferSchema", "true")
           .csv(CSV_PATH))

    # ---------- Step 2: PySpark processing ----------
    print("Step 2: cleaning and preparing columns")
    df = raw.withColumn(TARGET, F.trim(F.col(TARGET)))
    df = df.filter(F.col(TARGET).isin(list(LABEL_MAP)))
    df = df.withColumn("label",
                       F.when(F.col(TARGET) == "Low", 0.0)
                        .when(F.col(TARGET) == "Medium", 1.0)
                        .otherwise(2.0)).drop(TARGET)

    features = [c for c in df.columns if c not in ("label", *EXCLUDE_COLS)]
    numeric_cols = [f.name for f in df.schema.fields
                    if f.name in features and isinstance(f.dataType, NumericType)]
    string_cols = [c for c in features if c not in numeric_cols]

    distinct = {}
    if string_cols:
        row = df.agg(*[F.approx_count_distinct(c).alias(c) for c in string_cols]).first()
        distinct = row.asDict()
    cat_cols = [c for c in string_cols if distinct[c] <= MAX_CAT_LEVELS]
    dropped_cols = [c for c in string_cols if c not in cat_cols]

    for c in numeric_cols:
        df = df.withColumn(c, F.col(c).cast("double"))
    df = df.select(*numeric_cols, *cat_cols, "label")

    counts = {r["label"]: r["count"] for r in df.groupBy("label").count().collect()}
    run["rows_after_cleaning"] = int(sum(counts.values()))
    run["label_counts_before_balancing"] = {LABEL_NAMES[k]: int(v) for k, v in counts.items()}
    run["numeric_columns"] = numeric_cols
    run["categorical_columns"] = cat_cols
    run["dropped_text_columns"] = dropped_cols

    # Balance the classes by sampling up to PER_CLASS rows from each.
    fractions = {k: min(1.0, PER_CLASS / v) for k, v in counts.items()}
    balanced = df.sampleBy("label", fractions=fractions, seed=SEED).dropDuplicates().cache()
    balanced.count()

    train_raw, test_raw = balanced.randomSplit([0.8, 0.2], seed=SEED)
    run["test_rows_also_in_train"] = test_raw.intersect(train_raw).count()  # should be 0

    # ---------- Step 3: Spark ML preprocessing ----------
    print("Step 3: building preprocessing pipeline")
    stages = [Imputer(inputCols=numeric_cols,
                      outputCols=[f"{c}_imp" for c in numeric_cols])]
    if cat_cols:
        for c in cat_cols:
            stages.append(StringIndexer(inputCol=c, outputCol=f"{c}_idx",
                                        handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCols=[f"{c}_idx" for c in cat_cols],
                                    outputCols=[f"{c}_ohe" for c in cat_cols],
                                    handleInvalid="keep"))
    assembled = [f"{c}_imp" for c in numeric_cols] + [f"{c}_ohe" for c in cat_cols]
    stages.append(VectorAssembler(inputCols=assembled, outputCol="features_raw",
                                  handleInvalid="skip"))
    stages.append(StandardScaler(inputCol="features_raw", outputCol="features",
                                 withMean=False, withStd=True))
    prep_model = Pipeline(stages=stages).fit(train_raw)   # fitted on training data only

    train = prep_model.transform(train_raw).select("features", "label").cache()
    test = prep_model.transform(test_raw).select("features", "label").cache()
    run["train_rows"] = train.count()
    run["test_rows"] = test.count()

    # ---------- Step 4: Train and evaluate ----------
    print("Step 4: training and evaluating models")
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    models = {
        "LogisticRegression": LogisticRegression(maxIter=30, regParam=0.05,
                                                 family="multinomial"),
        "RandomForest": RandomForestClassifier(numTrees=30, maxDepth=8,
                                               subsamplingRate=0.7,
                                               featureSubsetStrategy="sqrt", seed=SEED),
    }

    metric_rows, fitted, confusion_tables = [], {}, {}
    for name, estimator in models.items():
        t0 = time.time()
        model = estimator.fit(train)
        pred = model.transform(test).cache()
        fitted[name] = model

        row = {"model": name,
               "accuracy": evaluator.evaluate(pred, {evaluator.metricName: "accuracy"}),
               "weighted_f1": evaluator.evaluate(pred, {evaluator.metricName: "f1"}),
               "weighted_precision": evaluator.evaluate(pred, {evaluator.metricName: "weightedPrecision"}),
               "weighted_recall": evaluator.evaluate(pred, {evaluator.metricName: "weightedRecall"}),
               "train_seconds": round(time.time() - t0, 2)}
        metric_rows.append(row)

        cm = pred.groupBy("label", "prediction").count().toPandas()
        confusion_tables[name] = cm
        cm.assign(true_class=cm["label"].map(LABEL_NAMES),
                  predicted_class=cm["prediction"].map(LABEL_NAMES)) \
          .to_csv(os.path.join(OUTPUT_DIR, f"confusion_{name}.csv"), index=False)
        pred.unpersist()

    # Per-class precision, recall and F1 from the confusion counts.
    per_class_rows = []
    for name, cm in confusion_tables.items():
        for code, cname in LABEL_NAMES.items():
            tp = cm[(cm.label == code) & (cm.prediction == code)]["count"].sum()
            support = cm[cm.label == code]["count"].sum()
            predicted = cm[cm.prediction == code]["count"].sum()
            p = tp / predicted if predicted else 0.0
            r = tp / support if support else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            per_class_rows.append({"model": name, "class": cname, "precision": round(p, 4),
                                   "recall": round(r, 4), "f1": round(f1, 4),
                                   "support": int(support)})

    # Random Forest feature importance mapped back to readable names.
    enc = next((s for s in prep_model.stages if isinstance(s, OneHotEncoderModel)), None)
    names = list(numeric_cols)
    if enc is not None:
        for c, size in zip(cat_cols, enc.categorySizes):
            names += [f"{c}_level_{i}" for i in range(max(0, size - 1))]
    importances = fitted["RandomForest"].featureImportances.toArray()
    fi = pd.DataFrame({"feature": names[:len(importances)],
                       "importance": importances[:len(names)]}) \
           .sort_values("importance", ascending=False)

    # ---------- Step 5: Save results ----------
    print("Step 5: saving results to", OUTPUT_DIR)
    pd.DataFrame(metric_rows).to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)
    pd.DataFrame(per_class_rows).to_csv(os.path.join(OUTPUT_DIR, "per_class_metrics.csv"), index=False)
    fi.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_random_forest.csv"), index=False)
    pd.DataFrame([{"label_code": k, "class": v} for k, v in LABEL_NAMES.items()]) \
      .to_csv(os.path.join(OUTPUT_DIR, "label_mapping.csv"), index=False)

    for name, model in fitted.items():
        model.write().overwrite().save(os.path.join(OUTPUT_DIR, "models", name))
    prep_model.write().overwrite().save(os.path.join(OUTPUT_DIR, "models", "preprocessing"))

    run["metrics"] = metric_rows
    run["total_seconds"] = round(time.time() - t_start, 1)
    run["finished"] = datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(OUTPUT_DIR, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, default=str)

    print("\nDone. Results saved in:", OUTPUT_DIR)
    print(pd.DataFrame(metric_rows).to_string(index=False))

finally:
    spark.stop()