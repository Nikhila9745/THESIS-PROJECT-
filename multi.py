# ============================================================
# STUDENT STRESS - ONE METHODOLOGY, MANY DATASETS (PySpark)
# ------------------------------------------------------------
# Runs the same methodology on every usable tabular dataset:
#   extract zips -> remove duplicates -> clean column names ->
#   map the target to Low / Medium / High -> engineer features ->
#   split 80/20 -> balance TRAINING data only -> preprocess (fit on
#   training only) -> Logistic Regression, Decision Tree, Naive Bayes,
#   MLP, Random Forest -> evaluate on the natural test distribution ->
#   repeat with several seeds (mean ± sd) -> importance + coefficients.
#
# Image datasets and the Dreaddit text dataset are NOT run here: they
# need CNN / NLP pipelines, not this tabular Spark ML methodology.
#
# Run (VS Code terminal, inside the folder with this file):
#   python multi_dataset_stress.py
#   python multi_dataset_stress.py --data-dir "D:\THESIS PROJECT" --repeats 5
#   python multi_dataset_stress.py --only stress_level_1100
# ============================================================

import os
import re
import sys
import csv
import glob
import json
import time
import zipfile
import hashlib
import unicodedata
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------
DATA_DIR = r"D:\THESIS PROJECT"                 # folder with your zip files
OUTPUT_DIR = r"D:\THESIS PROJECT\ML_multi_dataset_results"
JAVA_HOME = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"
SPARK_DRIVER_MEMORY = "6g"

REPEATS = 5                       # repeated random 80/20 splits -> mean ± sd
TEST_FRACTION = 0.2
TRAIN_ROWS_PER_CLASS = 10000      # cap; small datasets are limited by their smallest class
MAX_CATEGORIES = 30               # text columns with more distinct values are dropped
SEED = 42
CLASSES = ["Low", "Medium", "High"]
TOP_K = 10
MODELS = ["Logistic Regression", "Decision Tree", "Naive Bayes", "Multilayer Perceptron", "Random Forest"]

# Each dataset: which CSV, how to find the target, how to map it to Low/Medium/High,
# what to exclude (leakage / identifiers) and any column-splitting it needs.
#   "target"     : part of the column name (matched ignoring case, spaces, symbols)
#   "numeric"    : rules for numeric targets  -> list of ([values], label)
#   "text"       : rules for text targets     -> {value: label}
#   "exclude"    : cleaned column names to drop (leakage or identifiers)
#   "split_cols" : "120/80"-style text columns -> two numeric columns
DATASETS = [
    {
        "name": "survey_student_stress_factors",
        "files": ["Student Stress Factors (2).csv", "Student Stress Factors.csv"],
        "domain": "students - 1-5 Likert survey (sleep, headaches, academics, study load, activities)",
        "target": "rate your stress levels",
        "numeric": [([1, 2], "Low"), ([3], "Medium"), ([4, 5], "High")],
        "exclude": ["timestamp"],
        "notes": "second file is a later wave; appended only if it adds new rows",
    },
    {
        "name": "sleep_health_lifestyle",
        "files": ["Sleep_health_and_lifestyle_dataset.csv"],
        "domain": "adults incl. students - sleep, activity, BMI, blood pressure, heart rate",
        "target": "stress level",
        "numeric": [([1, 2, 3, 4], "Low"), ([5, 6], "Medium"), ([7, 8, 9, 10], "High")],
        "exclude": ["person_id"],
        "split_cols": {"blood_pressure": ("systolic", "diastolic")},
        "notes": "stress scale 1-10 mapped to 3 levels",
    },
    {
        "name": "stress_level_1100",
        "files": ["StressLevelDataset.csv"],
        "domain": "students - 20 factors (psychological, physiological, environmental, academic, social)",
        "target": "stress_level",
        "numeric": [([0], "Low"), ([1], "Medium"), ([2], "High")],
        "exclude": [],
        "notes": "base dataset of Omarbekova et al. (2026, dataset 2)",
    },
    {
        "name": "university_student_stress_3000",
        "files": ["university_student_stress_dataset.csv"],
        "domain": "university students - study habits, sleep, family, peers, anxiety",
        "target": "stress_level",
        "text": {"low": "Low", "medium": "Medium", "high": "High"},
        "exclude": ["stress_score"],   # Stress_Score was used to build the label -> leakage
        "notes": "cleaned file used; raw Google-Form export not used",
    },
    {
        "name": "stressed_213",
        "files": ["stressed.csv"],
        "domain": "students - categorical stressors (financial, family, academic, health ...)",
        "target": "stress_level",
        "text": {"normal": "Low", "stressed": "Medium", "highly stressed": "High"},
        "exclude": [],
        "notes": "multi-answer columns (separated by ';') become counts of issues",
    },
    {
        "name": "academic_stress_index_139",
        "files": ["academic Stress level - maintainance 1.csv"],
        "domain": "students - academic stage, peer and home pressure, study environment, coping",
        "target": "academic stress index",
        "numeric": [([1, 2], "Low"), ([3], "Medium"), ([4, 5], "High")],
        "exclude": ["timestamp"],
        "notes": "very small sample - treat results as indicative only",
    },
]

# Recorded for the thesis inventory (not run by this script)
NOT_RUN = [
    ("dataset_3__images_also_there_.zip", "image files (facial expressions)", "needs a CNN, not this tabular methodology"),
    ("dataset_4_images.zip", "image files (student engagement)", "needs a CNN, not this tabular methodology"),
    ("dataset_9__images_.zip", "nested archive (4).zip of engagement images", "needs a CNN, not this tabular methodology"),
    ("dreaddit-train.csv / dreaddit-test.csv (dataset 8)", "Reddit text posts, binary label", "needs NLP (TF-IDF / BERT), separate track"),
    ("mental_health_workplace_survey.csv (dataset 16)", "employees, not students; binary BurnoutRisk", "out-of-domain check - run separately"),
]


# ------------------------------------------------------------
# 2. ARGUMENTS
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--out", default=OUTPUT_DIR)
    p.add_argument("--repeats", type=int, default=REPEATS)
    p.add_argument("--only", nargs="*", help="run only these dataset names")
    p.add_argument("--driver-memory", default=SPARK_DRIVER_MEMORY)
    return p.parse_args()


# ------------------------------------------------------------
# 3. FILE HANDLING (extract zips, find CSVs, remove duplicates)
# ------------------------------------------------------------
def extract_zips(data_dir, work_dir):
    os.makedirs(work_dir, exist_ok=True)
    for z in glob.glob(os.path.join(data_dir, "*.zip")):
        dest = os.path.join(work_dir, os.path.splitext(os.path.basename(z))[0])
        if not os.path.isdir(dest):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest)
    return work_dir


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_csvs(roots):
    found = []
    for root in roots:
        found += glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    return found


# ------------------------------------------------------------
# 4. COLUMN NAMES AND TARGET RULES
# ------------------------------------------------------------
def clean_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    return s or "col"


def unique_names(names):
    seen, out = {}, []
    for n in names:
        base = clean_name(n)
        k, i = base, 2
        while k in seen:
            k, i = f"{base}_{i}", i + 1
        seen[k] = True
        out.append(k)
    return out


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def find_target(cols, key):
    hits = [c for c in cols if norm(key) in norm(c)]
    if len(hits) != 1:
        raise ValueError(f"target '{key}' matched {hits} - check the DATASETS config")
    return hits[0]


def is_id_column(c):
    return c in ("id", "record_id", "person_id", "student_id", "employee_id") or c.endswith("_id") or c.startswith("id_")


# ------------------------------------------------------------
# 5. SPARK PREPARATION
# ------------------------------------------------------------
def build_spark(args):
    from pyspark.sql import SparkSession
    if os.path.isdir(JAVA_HOME):
        os.environ["JAVA_HOME"] = JAVA_HOME
        os.environ["PATH"] = os.path.join(JAVA_HOME, "bin") + os.pathsep + os.environ["PATH"]
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    spark = (SparkSession.builder.appName("StudentStress_MultiDataset")
             .master("local[*]")
             .config("spark.driver.memory", args.driver_memory)
             .config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark


def read_csv(spark, path):
    df = (spark.read.option("header", "true").option("inferSchema", "true")
          .option("mode", "PERMISSIVE").csv(path))
    return df.toDF(*unique_names(df.columns))


def load_dataset(spark, cfg, paths, log):
    """Read, union the files of one dataset, map the target, and engineer features."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import NumericType, StringType

    base = read_csv(spark, paths[0])
    rows_per_file = {os.path.basename(paths[0]): base.count()}
    for extra_path in paths[1:]:
        extra = read_csv(spark, extra_path)
        keep = [c for c in base.columns if c in extra.columns and c not in cfg["exclude"]]
        new_rows = extra.select(keep).exceptAll(base.select(keep)).count()
        rows_per_file[os.path.basename(extra_path)] = extra.count()
        if new_rows == 0:
            log.append(f"{os.path.basename(extra_path)}: all rows already present -> not appended")
            continue
        log.append(f"{os.path.basename(extra_path)}: {new_rows} rows not in first file -> appended")
        base = base.select(keep).unionByName(extra.select(keep))
    df = base

    target = find_target(df.columns, cfg["target"])
    if "numeric" in cfg:
        d = F.col(target).cast("double")
        expr = None
        for values, label in cfg["numeric"]:
            cond = d.isin([float(v) for v in values])
            expr = F.when(cond, F.lit(label)) if expr is None else expr.when(cond, F.lit(label))
    else:
        s = F.lower(F.trim(F.col(target).cast("string")))
        expr = None
        for value, label in cfg["text"].items():
            cond = s == value
            expr = F.when(cond, F.lit(label)) if expr is None else expr.when(cond, F.lit(label))
    df = df.withColumn("label_name", expr.otherwise(F.lit(None).cast("string")))
    before = df.count()
    df = df.filter(F.col("label_name").isNotNull())
    dropped_labels = before - df.count()
    if dropped_labels:
        log.append(f"{dropped_labels} rows had a target value outside the mapping and were dropped")

    df = df.withColumn("label", F.when(F.col("label_name") == CLASSES[0], 0.0)
                       .when(F.col("label_name") == CLASSES[1], 1.0)
                       .when(F.col("label_name") == CLASSES[2], 2.0))

    # "120/80" style columns -> two numeric columns
    for orig, (a, b) in cfg.get("split_cols", {}).items():
        if orig in df.columns:
            df = (df.withColumn(f"{orig}_{a}", F.regexp_extract(F.col(orig), r"^\s*(\d+)\s*/\s*(\d+)\s*$", 1).cast("double"))
                    .withColumn(f"{orig}_{b}", F.regexp_extract(F.col(orig), r"^\s*(\d+)\s*/\s*(\d+)\s*$", 2).cast("double"))
                    .drop(orig))
            log.append(f"'{orig}' split into {orig}_{a} and {orig}_{b}")

    excluded = set(cfg["exclude"]) | {target, "label_name", "label"}
    # multi-answer text columns ("a;b;None") -> number of real answers
    for c in list(df.columns):
        if c in excluded or is_id_column(c):
            continue
        if dict(df.dtypes)[c] == "string" and df.filter(F.col(c).contains(";")).limit(1).count() > 0:
            items = F.split(F.col(c), ";")
            cnt = F.size(F.filter(items, lambda x: ~F.lower(F.trim(x)).isin("none", "", "no")))
            df = df.withColumn(f"{c}_count", F.coalesce(cnt, F.lit(0)).cast("double")).drop(c)
            log.append(f"'{c}' is multi-answer -> '{c}_count'")
            excluded.add(c)

    feats = [c for c in df.columns if c not in excluded and not is_id_column(c)]
    types = dict(df.dtypes)
    numeric, categorical, dropped = [], [], []
    n_rows = df.count()
    for c in feats:
        t = types[c]
        if t in ("int", "bigint", "double", "float", "smallint", "tinyint", "decimal") or t.startswith("decimal"):
            if df.filter(F.col(c).isNotNull()).limit(1).count() == 0:
                dropped.append(f"{c} (all empty)")
                continue
            numeric.append(c)
        elif t == "string":
            k = df.select(F.countDistinct(F.col(c)).alias("k")).collect()[0]["k"]
            if k > MAX_CATEGORIES:
                dropped.append(f"{c} ({k} categories)")
            else:
                categorical.append(c)
        else:
            dropped.append(f"{c} ({t})")
    for c in numeric:
        df = df.withColumn(c, F.col(c).cast("double"))
    if categorical:
        df = df.fillna("Unknown", subset=categorical)
    keep = numeric + categorical + ["label"]
    df = df.select(*keep)
    return df, {
        "target_column": target, "rows_after_mapping": n_rows,
        "files_rows": rows_per_file, "numeric_features": numeric,
        "categorical_features": categorical, "dropped_columns": dropped,
    }


# ------------------------------------------------------------
# 6. ML HELPERS
# ------------------------------------------------------------
def exact_sample(df, n, seed):
    from pyspark.sql import Window, functions as F
    w = Window.partitionBy("label").orderBy(F.rand(seed))
    return (df.withColumn("__rn", F.row_number().over(w))
              .filter(F.col("__rn") <= n).drop("__rn"))


def build_preprocessing(numeric, categorical):
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
    stages, imp, enc = [], [], []
    if numeric:
        imp = [c + "_imputed" for c in numeric]
        stages.append(Imputer(inputCols=numeric, outputCols=imp, strategy="median"))
    if categorical:
        idx = [c + "_indexed" for c in categorical]
        enc = [c + "_encoded" for c in categorical]
        stages.append(StringIndexer(inputCols=categorical, outputCols=idx, handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCols=idx, outputCols=enc, handleInvalid="keep"))
    stages.append(VectorAssembler(inputCols=imp + enc, outputCol="raw_features", handleInvalid="keep"))
    stages.append(StandardScaler(inputCol="raw_features", outputCol="features", withMean=False, withStd=True))
    return Pipeline(stages=stages)


def feature_names_from(df):
    attrs = df.schema["raw_features"].metadata.get("ml_attr", {}).get("attrs", {})
    names = {}
    for group in attrs.values():
        for a in group:
            names[int(a["idx"])] = a["name"]
    return names


def source_of(encoded, numeric, categorical):
    if encoded.endswith("_imputed") and encoded[:-8] in numeric:
        return encoded[:-8]
    for c in sorted(categorical, key=len, reverse=True):
        if encoded.startswith(c + "_encoded"):
            return c
    return encoded


def confusion(pred_df, k=3):
    m = np.zeros((k, k), dtype=np.int64)
    for r in pred_df.groupBy("label", "prediction").count().collect():
        m[int(r["label"]), int(r["prediction"])] += int(r["count"])
    return m


def metrics_from_cm(M):
    M = M.astype(float)
    k = M.shape[0]
    total = M.sum()
    support, predicted, tp = M.sum(axis=1), M.sum(axis=0), np.diag(M)
    prec = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
    rec = np.divide(tp, support, out=np.zeros(k), where=support > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    w = support / total if total else np.zeros(k)
    acc = tp.sum() / total if total else 0.0
    pe = (support * predicted).sum() / (total * total) if total else 0.0
    kappa = (acc - pe) / (1 - pe) if pe < 1 else 0.0
    num = tp.sum() * total - (predicted * support).sum()
    den = np.sqrt((total ** 2 - (predicted ** 2).sum()) * (total ** 2 - (support ** 2).sum()))
    mcc = num / den if den > 0 else 0.0
    return {
        "accuracy": float(acc), "weighted_f1": float((f1 * w).sum()),
        "macro_precision": float(prec.mean()), "macro_recall": float(rec.mean()),
        "macro_f1": float(f1.mean()), "balanced_accuracy": float(rec.mean()),
        "mcc": float(mcc), "kappa": float(kappa),
        "recall_Low": float(rec[0]), "recall_Medium": float(rec[1]), "recall_High": float(rec[2]),
    }


def mlp_diagnostics(model):
    """Optimizer health from the TRAINING objective history only (never the test set)."""
    try:
        s = model.summary
        s = s() if callable(s) else s
        hist = [float(v) for v in s.objectiveHistory]
    except Exception:
        return {"mlp_iterations": None, "mlp_final_loss": None, "mlp_nonfinite": None, "mlp_stalled": None}
    finite = [v for v in hist if np.isfinite(v)]
    tail = finite[-10:]
    stalled = len(tail) >= 2 and (tail[0] - tail[-1]) <= 1e-6 * max(abs(tail[0]), 1e-12)
    return {"mlp_iterations": len(hist), "mlp_final_loss": finite[-1] if finite else None,
            "mlp_nonfinite": len(hist) - len(finite), "mlp_stalled": bool(stalled)}


# ------------------------------------------------------------
# 7. RUN ONE DATASET
# ------------------------------------------------------------
def run_dataset(spark, cfg, paths, out_dir, repeats, log):
    from pyspark import StorageLevel
    from pyspark.sql import functions as F
    from pyspark.ml.feature import MinMaxScaler
    from pyspark.ml.functions import vector_to_array, array_to_vector
    from pyspark.ml.classification import (LogisticRegression, DecisionTreeClassifier, NaiveBayes,
                                           MultilayerPerceptronClassifier, RandomForestClassifier)

    os.makedirs(out_dir, exist_ok=True)
    df, info = load_dataset(spark, cfg, paths, log)
    df = df.persist(StorageLevel.MEMORY_AND_DISK)
    dist = {r["label"]: int(r["count"]) for r in df.groupBy("label").count().collect()}
    if len(dist) < 3:
        raise ValueError(f"only {len(dist)} classes present after mapping - need Low, Medium and High")
    info["class_distribution"] = {CLASSES[int(k)]: v for k, v in sorted(dist.items())}
    numeric, categorical = info["numeric_features"], info["categorical_features"]
    print(f"  rows={info['rows_after_mapping']:,}  classes={info['class_distribution']}  "
          f"numeric={len(numeric)} categorical={len(categorical)}")

    rows, cm_sum, importance_rows, coef_rows = [], {m: np.zeros((3, 3), dtype=np.int64) for m in MODELS}, [], []
    for r in range(repeats):
        seed = SEED + r
        train_all, test = df.randomSplit([1 - TEST_FRACTION, TEST_FRACTION], seed=seed)
        counts = {int(x["label"]): int(x["count"]) for x in train_all.groupBy("label").count().collect()}
        n_per_class = min(TRAIN_ROWS_PER_CLASS, min(counts.values()))
        train = exact_sample(train_all, n_per_class, seed).persist(StorageLevel.MEMORY_AND_DISK)

        prep = build_preprocessing(numeric, categorical).fit(train)
        train_p = prep.transform(train)
        names = feature_names_from(train_p)
        nb_scaler = MinMaxScaler(inputCol="raw_features", outputCol="nb_features").fit(train_p)
        nb_clip = array_to_vector(F.transform(vector_to_array("nb_features"),
                                              lambda x: F.least(F.greatest(x, F.lit(0.0)), F.lit(1.0))))
        train_p = nb_scaler.transform(train_p).withColumn("nb_features", nb_clip) \
            .select("label", "features", "nb_features", "raw_features").persist(StorageLevel.MEMORY_AND_DISK)
        test_p = nb_scaler.transform(prep.transform(test)).withColumn("nb_features", nb_clip) \
            .select("label", "features", "nb_features", "raw_features").persist(StorageLevel.MEMORY_AND_DISK)
        n_features = len(train_p.first()["features"])
        n_test = test_p.count()

        specs = [
            ("Logistic Regression", LogisticRegression(featuresCol="features", labelCol="label",
                                                       maxIter=100, regParam=0.01, family="multinomial"), "features"),
            ("Decision Tree", DecisionTreeClassifier(featuresCol="features", labelCol="label",
                                                     maxDepth=6, seed=seed), "features"),
            ("Naive Bayes", NaiveBayes(featuresCol="nb_features", labelCol="label",
                                       smoothing=1.0, modelType="multinomial"), "nb_features"),
            ("Multilayer Perceptron", MultilayerPerceptronClassifier(
                featuresCol="features", labelCol="label", layers=[n_features, 32, 16, 3],
                maxIter=1000, blockSize=64, seed=seed), "features"),  # raised from 200: survey MLP was still improving at the cap
            ("Random Forest", RandomForestClassifier(featuresCol="features", labelCol="label",
                                                     numTrees=100, maxDepth=8,
                                                     featureSubsetStrategy="sqrt", seed=seed), "features"),
        ]
        for name, clf, _ in specs:
            t0 = time.time()
            # MLP: one partition so the gradient summation order (and so the result) does not depend on core count
            fitted = clf.fit(train_p.coalesce(1) if name == "Multilayer Perceptron" else train_p)
            train_sec = time.time() - t0
            M = confusion(fitted.transform(test_p))
            cm_sum[name] += M
            m = metrics_from_cm(M)
            diag = mlp_diagnostics(fitted) if name == "Multilayer Perceptron" else {}
            rows.append({"dataset": cfg["name"], "repeat": r, "seed": seed, "model": name,
                         "train_rows": n_per_class * 3, "test_rows": n_test,
                         "train_seconds": round(train_sec, 2), **m, **diag})
            if diag:
                print(f"    repeat {r+1}/{repeats}  MLP diagnostics: iters={diag['mlp_iterations']} "
                      f"final_loss={diag['mlp_final_loss']} nonfinite={diag['mlp_nonfinite']} "
                      f"stalled={diag['mlp_stalled']}")
            if name == "Random Forest" or name == "Decision Tree":
                imp = fitted.featureImportances.toArray()
                for idx, val in enumerate(imp):
                    enc = names.get(idx, f"f{idx}")
                    importance_rows.append({"dataset": cfg["name"], "repeat": r, "model": name, "feature": enc,
                                            "attribute": source_of(enc, numeric, categorical), "importance": float(val)})
            if name == "Logistic Regression" and r == 0:
                W = fitted.coefficientMatrix.toArray()
                for idx in range(W.shape[1]):
                    enc = names.get(idx, f"f{idx}")
                    coef_rows.append({"dataset": cfg["name"], "feature": enc,
                                      "attribute": source_of(enc, numeric, categorical),
                                      **{f"coef_{CLASSES[k]}": float(W[k][idx]) for k in range(3)}})
            print(f"    repeat {r+1}/{repeats}  {name:<22} acc={m['accuracy']:.3f}  macroF1={m['macro_f1']:.3f}  "
                  f"MCC={m['mcc']:.3f}  ({train_sec:.1f}s)")
        for d in (train, train_p, test_p):
            d.unpersist()

    df.unpersist()
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "metrics_by_repeat.csv"), index=False)
    summary = res.groupby("model").agg(
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        weighted_f1_mean=("weighted_f1", "mean"), balanced_acc_mean=("balanced_accuracy", "mean"),
        mcc_mean=("mcc", "mean"), mcc_sd=("mcc", "std"),
        recall_Low_mean=("recall_Low", "mean"), recall_Medium_mean=("recall_Medium", "mean"),
        recall_High_mean=("recall_High", "mean"), train_seconds=("train_seconds", "mean"),
    ).reindex(MODELS).reset_index()
    summary.insert(0, "dataset", cfg["name"])
    summary.to_csv(os.path.join(out_dir, "model_summary.csv"), index=False)

    attr_imp = None
    if importance_rows:
        imp = pd.DataFrame(importance_rows)
        # one attribute's importance = sum over its one-hot levels, per repeat; then mean over repeats
        attr_imp = imp.groupby(["model", "repeat", "attribute"])["importance"].sum() \
            .groupby(["model", "attribute"]).mean().reset_index() \
            .sort_values(["model", "importance"], ascending=[True, False])
        attr_imp.to_csv(os.path.join(out_dir, "feature_importance_by_attribute.csv"), index=False)
    if coef_rows:
        pd.DataFrame(coef_rows).to_csv(os.path.join(out_dir, "logistic_regression_coefficients.csv"), index=False)

    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({**info, "name": cfg["name"], "domain": cfg["domain"], "notes": cfg["notes"],
                   "repeats": repeats, "train_rows_per_class": n_per_class,
                   "cleaning_log": log}, f, indent=2, default=str)
    plot_dataset(cfg["name"], out_dir, summary, cm_sum, attr_imp)
    return summary, info


# ------------------------------------------------------------
# 8. PLOTS
# ------------------------------------------------------------
def plot_dataset(name, out_dir, summary, cm_sum, imp):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(summary))
    ax.bar(x - 0.2, summary.accuracy_mean, 0.4, yerr=summary.accuracy_sd.fillna(0), capsize=3, label="Accuracy")
    ax.bar(x + 0.2, summary.macro_f1_mean, 0.4, yerr=summary.macro_f1_sd.fillna(0), capsize=3, label="Macro-F1")
    ax.set_xticks(x, summary.model, rotation=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{name}: model performance (mean ± sd over repeated splits)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.6 * len(MODELS), 3.8))
    for ax, m in zip(axes, MODELS):
        M = cm_sum[m].astype(float)
        norm_m = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm_m, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{int(M[i, j]):,}", ha="center", va="center", fontsize=7,
                        color="white" if norm_m[i, j] > 0.5 else "black")
        ax.set_xticks(range(3), CLASSES, fontsize=8)
        ax.set_yticks(range(3), CLASSES, fontsize=8)
        ax.set_title(m, fontsize=9)
    axes[0].set_ylabel("Actual")
    fig.suptitle(f"{name}: confusion matrices summed over repeats")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    if imp is not None:
        rf = imp[imp.model == "Random Forest"].sort_values("importance", ascending=False).head(TOP_K)[::-1]
        if len(rf):
            fig, ax = plt.subplots(figsize=(8, 0.4 * len(rf) + 1.5))
            ax.barh([a.replace("_", " ") for a in rf.attribute], rf.importance.values, color="#264653")
            ax.set_title(f"{name}: Random Forest importance (top {TOP_K})")
            ax.set_xlabel("mean over repeats (attributes sum to 1)")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "rf_feature_importance.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)


def plot_overall(all_summary, out_dir):
    for metric, title in [("macro_f1_mean", "Macro-F1"), ("accuracy_mean", "Accuracy")]:
        piv = all_summary.pivot(index="dataset", columns="model", values=metric).reindex(columns=MODELS)
        fig, ax = plt.subplots(figsize=(11, 0.6 * len(piv) + 2))
        im = ax.imshow(piv.values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(j, i, "" if pd.isna(v) else f"{v:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=15, fontsize=8)
        ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=8)
        fig.colorbar(im, ax=ax, label=title)
        ax.set_title(f"{title} by dataset and model (mean over repeats)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"overall_{metric}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------------------------
# 9. MAIN
# ------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    work = os.path.join(args.data_dir, "_extracted")
    try:
        extract_zips(args.data_dir, work)
    except PermissionError:
        work = os.path.join(args.out, "_extracted")
        extract_zips(args.data_dir, work)

    # ---- inventory: every CSV found, its checksum, and whether it was used
    csvs = find_csvs([work, args.data_dir])
    study_files = {f for d in DATASETS for f in d["files"]}
    inventory, seen = [], {}
    for p in sorted(set(c for c in csvs if os.path.basename(c) in study_files)):
        h = md5(p)
        status = "duplicate of " + seen[h] if h in seen else "unique"
        seen.setdefault(h, os.path.basename(p))
        inventory.append({"file": os.path.basename(p), "md5": h, "status": status})

    selected = [d for d in DATASETS if not args.only or d["name"] in args.only]
    spark = build_spark(args)
    all_summaries, profiles = [], []
    try:
        for cfg in selected:
            print("\n" + "=" * 70 + f"\nDATASET: {cfg['name']}\n" + "=" * 70)
            paths, used_hashes = [], set()
            for fname in cfg["files"]:
                hits = [p for p in csvs if os.path.basename(p) == fname]
                if not hits:
                    raise FileNotFoundError(f"'{fname}' not found under {args.data_dir}")
                p = hits[0]
                h = md5(p)
                if h in used_hashes:
                    print(f"  skipped identical copy: {p}")
                    continue
                used_hashes.add(h)
                paths.append(p)
            log = []
            out_dir = os.path.join(args.out, cfg["name"])
            t0 = time.time()
            summary, info = run_dataset(spark, cfg, paths, out_dir, args.repeats, log)
            info["seconds"] = round(time.time() - t0, 1)
            all_summaries.append(summary)
            profiles.append({"dataset": cfg["name"], "domain": cfg["domain"], "files_used": "; ".join(os.path.basename(p) for p in paths),
                             "rows": info["rows_after_mapping"], "target_column": info["target_column"],
                             "class_counts": json.dumps(info["class_distribution"]),
                             "numeric_features": len(info["numeric_features"]),
                             "categorical_features": len(info["categorical_features"]),
                             "dropped": "; ".join(info["dropped_columns"]), "notes": cfg["notes"],
                             "cleaning_log": " | ".join(log), "seconds": info["seconds"]})
    finally:
        spark.stop()

    pd.DataFrame(inventory).to_csv(os.path.join(args.out, "file_inventory.csv"), index=False)
    pd.DataFrame(profiles).to_csv(os.path.join(args.out, "dataset_profiles.csv"), index=False)
    pd.DataFrame(NOT_RUN, columns=["file", "content", "reason_not_run"]).to_csv(
        os.path.join(args.out, "not_run_datasets.csv"), index=False)
    if all_summaries:
        allS = pd.concat(all_summaries, ignore_index=True)
        allS.to_csv(os.path.join(args.out, "all_datasets_model_summary.csv"), index=False)
        plot_overall(allS, args.out)
        print("\n=== Macro-F1 (mean) by dataset ===")
        print(allS.pivot(index="dataset", columns="model", values="macro_f1_mean")
              .reindex(columns=MODELS).round(3).to_string())
    print(f"\nAll results saved in: {args.out}")


if __name__ == "__main__":
    main()