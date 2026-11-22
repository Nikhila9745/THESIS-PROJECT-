# methods_suite.py
# Runs the full set of checks on one dataset:
#   1. Repeated splits (5 seeds) for 7 classifiers plus ordinal regression, with mean and SD
#   2. Bootstrap 95% confidence intervals on the test set (seed 42)
#   3. McNemar tests between every pair of models (seed 42)
#   4. Learning curve (share of training data used)
#   5. Feature-group ablation (remove one attribute group, retrain)
#   6. Label-derivation check (shallow trees on Stress_Score and on the other features)
# The same script works for any size: pass the CSV and output folder.
#
# Run from the Ubuntu terminal with the sparkenv active, for example:
#   python3 methods_suite.py --csv "/mnt/d/THESIS PROJECT/Stress_Data_5GB.csv" --out "/mnt/c/Users/JP/THESIS PROJECT/methods_5GB"

import os
import json
import math
import time
import argparse
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import NumericType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (Imputer, StringIndexer, OneHotEncoder, VectorAssembler,
                                StandardScaler, MinMaxScaler)
from pyspark.ml.classification import (LogisticRegression, RandomForestClassifier,
                                       DecisionTreeClassifier, MultilayerPerceptronClassifier,
                                       NaiveBayes, LinearSVC, OneVsRest, GBTClassifier)
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--seeds", type=int, default=5)
parser.add_argument("--per-class", type=int, default=10000)
parser.add_argument("--bootstrap", type=int, default=1000)
parser.add_argument("--driver-memory", default="6g")
args = parser.parse_args()

TARGET = "Stress_Level"
LABEL_MAP = {"Low": 0.0, "Medium": 1.0, "High": 2.0}
LABEL_NAMES = ["Low", "Medium", "High"]
EXCLUDE_COLS = ["Record_ID", "Stress_Score", "Timestamp"]
MAX_CAT_LEVELS = 50
BASE_SEED = 42

GROUPS = {
    "physiological": ["Heart_Rate", "HRV", "Systolic_BP", "Diastolic_BP", "Respiration_Rate", "Body_Temperature"],
    "mental_and_social": ["Anxiety_Level", "Mood_Score", "Financial_Stress", "Family_Stress",
                          "Relationship_Stress", "Social_Support"],
    "academic_workload": ["Study_Hours", "Working_Hours", "Academic_Workload", "Exam_Pressure",
                          "Assignment_Pressure", "Attendance_Percentage"],
    "lifestyle": ["Age", "BMI", "Sleep_Hours", "Screen_Time_Hours", "Social_Media_Hours",
                  "Exercise_Minutes", "Water_Intake_Liters"],
    "environment": ["Noise_Level", "Air_Quality_Index"],
    "categorical": ["Gender", "Education", "Occupation", "Marital_Status", "Diet_Quality",
                    "Smoking", "Alcohol", "Weather", "Activity", "Comment"],
}

os.makedirs(args.out, exist_ok=True)
spark = (SparkSession.builder.appName("Methods suite")
         .master("local[*]")
         .config("spark.driver.memory", args.driver_memory)
         .config("spark.sql.shuffle.partitions", "16")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")
t_start = time.time()
run = {"csv": args.csv, "seeds": args.seeds, "per_class": args.per_class, "bootstrap": args.bootstrap}


def save(frame, name):
    frame.to_csv(os.path.join(args.out, name), index=False)
    print("Saved", name)


def build_prep_stages(num_cols, cat_cols, scaler):
    stages = [Imputer(inputCols=num_cols, outputCols=[c + "_imp" for c in num_cols])]
    for c in cat_cols:
        stages.append(StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep"))
    if cat_cols:
        stages.append(OneHotEncoder(inputCols=[c + "_idx" for c in cat_cols],
                                    outputCols=[c + "_ohe" for c in cat_cols], handleInvalid="keep"))
    stages.append(VectorAssembler(inputCols=[c + "_imp" for c in num_cols] + [c + "_ohe" for c in cat_cols],
                                  outputCol="features_raw", handleInvalid="skip"))
    if scaler == "minmax":
        stages.append(MinMaxScaler(inputCol="features_raw", outputCol="features"))
    else:
        stages.append(StandardScaler(inputCol="features_raw", outputCol="features",
                                     withMean=False, withStd=True))
    return stages


from pyspark.ml.linalg import Vectors, VectorUDT


def clip_unit(v):
    # MinMax scaling fitted on the training split can put test values outside [0, 1].
    # Naive Bayes needs non-negative inputs, so clip them to the [0, 1] range.
    return Vectors.dense([min(max(x, 0.0), 1.0) for x in v.toArray()])


clip_udf = F.udf(clip_unit, VectorUDT())


def prepare(train, test, num_cols, cat_cols, scaler):
    model = Pipeline(stages=build_prep_stages(num_cols, cat_cols, scaler)).fit(train)
    tr = model.transform(train)
    te = model.transform(test).withColumn("rid", F.monotonically_increasing_id())
    if scaler == "minmax":
        tr = tr.withColumn("features", clip_udf("features"))
        te = te.withColumn("features", clip_udf("features"))
    tr = tr.select("features", "label").cache()
    te = te.select("rid", "features", "label").cache()
    return tr, te


def mcc(y, p, k=3):
    C = np.zeros((k, k))
    np.add.at(C, (y.astype(int), p.astype(int)), 1)
    s = C.sum()
    c = np.trace(C)
    t_k = C.sum(axis=1)
    p_k = C.sum(axis=0)
    den = math.sqrt((s * s - np.dot(p_k, p_k)) * (s * s - np.dot(t_k, t_k)))
    return float((c * s - np.dot(t_k, p_k)) / den) if den else 0.0


def weighted_f1(y, p, k=3):
    total = len(y)
    score = 0.0
    for c in range(k):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        support = np.sum(y == c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        score += f1 * support / total
    return float(score)


def make_models(n_features, seed):
    return {
        "Logistic Regression": (LogisticRegression(maxIter=30, regParam=0.05, family="multinomial"), "standard"),
        "Random Forest": (RandomForestClassifier(numTrees=30, maxDepth=8, subsamplingRate=0.7,
                                                 featureSubsetStrategy="sqrt", seed=seed), "standard"),
        "Decision Tree": (DecisionTreeClassifier(maxDepth=8, seed=seed), "standard"),
        "Gradient Boosted Trees (one-vs-rest)": (OneVsRest(classifier=GBTClassifier(maxIter=30, maxDepth=5, seed=seed)), "standard"),
        "Multilayer Perceptron": (MultilayerPerceptronClassifier(layers=[n_features, 32, 3], maxIter=100,
                                                                  blockSize=128, seed=seed), "standard"),
        "Naive Bayes": (NaiveBayes(modelType="multinomial"), "minmax"),
        "Linear SVM (one-vs-rest)": (OneVsRest(classifier=LinearSVC(maxIter=50, regParam=0.05)), "standard"),
    }


def predictions_frame(pred_df, label_col="label", pred_col="prediction"):
    return pred_df.select("rid", F.col(label_col).alias("y"), F.col(pred_col).alias("p")).toPandas()


# ------------------------------------------------------------------
# Load and clean
# ------------------------------------------------------------------
print("Loading and cleaning data")
raw = spark.read.option("header", "true").option("inferSchema", "true").csv(args.csv)
df = raw.withColumn(TARGET, F.trim(F.col(TARGET))).filter(F.col(TARGET).isin(list(LABEL_MAP)))
df = df.withColumn("label", F.when(F.col(TARGET) == "Low", 0.0)
                             .when(F.col(TARGET) == "Medium", 1.0).otherwise(2.0))
has_score = "Stress_Score" in df.columns
features = [c for c in df.columns if c not in ("label", TARGET, *EXCLUDE_COLS)]
numeric = [f.name for f in df.schema.fields if f.name in features and isinstance(f.dataType, NumericType)]
strings = [c for c in features if c not in numeric]
distinct = df.agg(*[F.approx_count_distinct(c).alias(c) for c in strings]).first().asDict() if strings else {}
cats = [c for c in strings if distinct[c] <= MAX_CAT_LEVELS]
for c in numeric + (["Stress_Score"] if has_score else []):
    df = df.withColumn(c, F.col(c).cast("double"))
numeric_with_score = numeric + (["Stress_Score"] if has_score else [])

counts = {r["label"]: r["count"] for r in df.groupBy("label").count().collect()}
run["label_counts_all"] = {LABEL_NAMES[int(k)]: int(v) for k, v in counts.items()}
run["numeric_predictors"] = numeric
run["categorical_predictors"] = cats

fractions = {k: min(1.0, args.per_class / v) for k, v in counts.items()}
keep_cols = list(dict.fromkeys(numeric_with_score + cats + ["label"]))
balanced = (df.select(*keep_cols)
              .sampleBy("label", fractions=fractions, seed=BASE_SEED)
              .dropDuplicates()
              .cache())
balanced_n = balanced.count()
run["balanced_records"] = balanced_n
print("Balanced records:", balanced_n)

# ------------------------------------------------------------------
# 1. Repeated splits: all models
# ------------------------------------------------------------------
print("\n1. Repeated splits")
rep_rows = []
seed42_preds = {}
for seed in range(BASE_SEED, BASE_SEED + args.seeds):
    train_raw, test_raw = balanced.randomSplit([0.8, 0.2], seed=seed)
    train_raw = train_raw.cache()
    test_raw = test_raw.cache()
    prep_cache = {}
    for scaler in ["standard", "minmax"]:
        prep_cache[scaler] = prepare(train_raw, test_raw, numeric, cats, scaler)
    n_features = prep_cache["standard"][0].first()["features"].size

    for name, (est, scaler) in make_models(n_features, seed).items():
        tr, te = prep_cache[scaler]
        t0 = time.time()
        model = est.fit(tr)
        train_seconds = time.time() - t0
        pred = model.transform(te)
        pdf = predictions_frame(pred)
        y, p = pdf["y"].values, pdf["p"].values
        rep_rows.append({"seed": seed, "model": name, "accuracy": float(np.mean(y == p)),
                         "weighted_f1": weighted_f1(y, p), "mcc": mcc(y, p),
                         "mae": float(np.mean(np.abs(y - p))), "train_seconds": round(train_seconds, 2)})
        print(f"  seed {seed} {name}: acc {rep_rows[-1]['accuracy']:.4f}")
        if seed == BASE_SEED:
            seed42_preds[name] = pdf

    # Ordinal regression: predict the ordered code, round and clip to 0..2
    tr, te = prep_cache["standard"]
    t0 = time.time()
    reg = LinearRegression(featuresCol="features", labelCol="label", maxIter=50, regParam=0.05).fit(tr)
    train_seconds = time.time() - t0
    pdf = predictions_frame(reg.transform(te))
    y = pdf["y"].values
    p = np.clip(np.round(pdf["p"].values), 0, 2)
    rep_rows.append({"seed": seed, "model": "Ordinal regression (rounded)",
                     "accuracy": float(np.mean(y == p)), "weighted_f1": weighted_f1(y, p),
                     "mcc": mcc(y, p), "mae": float(np.mean(np.abs(y - p))),
                     "train_seconds": round(train_seconds, 2)})
    if seed == BASE_SEED:
        pdf["p"] = p
        seed42_preds["Ordinal regression (rounded)"] = pdf

rep = pd.DataFrame(rep_rows)
save(rep, "repeated_seed_metrics.csv")
summary = rep.groupby("model").agg(
    accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
    weighted_f1_mean=("weighted_f1", "mean"), weighted_f1_sd=("weighted_f1", "std"),
    mcc_mean=("mcc", "mean"), mcc_sd=("mcc", "std"),
    mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
    train_seconds_mean=("train_seconds", "mean")).reset_index()
save(summary.sort_values("accuracy_mean", ascending=False), "repeated_summary.csv")

# ------------------------------------------------------------------
# 2. Bootstrap confidence intervals (seed 42 test set)
# ------------------------------------------------------------------
print("\n2. Bootstrap confidence intervals")
rng = np.random.default_rng(BASE_SEED)
boot_rows = []
for name, pdf in seed42_preds.items():
    y, p = pdf["y"].values, pdf["p"].values
    n = len(y)
    accs, f1s = [], []
    for _ in range(args.bootstrap):
        idx = rng.integers(0, n, n)
        accs.append(np.mean(y[idx] == p[idx]))
        f1s.append(weighted_f1(y[idx], p[idx]))
    boot_rows.append({"model": name,
                      "accuracy": float(np.mean(y == p)),
                      "accuracy_ci_low": float(np.percentile(accs, 2.5)),
                      "accuracy_ci_high": float(np.percentile(accs, 97.5)),
                      "weighted_f1": weighted_f1(y, p),
                      "weighted_f1_ci_low": float(np.percentile(f1s, 2.5)),
                      "weighted_f1_ci_high": float(np.percentile(f1s, 97.5))})
save(pd.DataFrame(boot_rows), "bootstrap_ci_seed42.csv")

# ------------------------------------------------------------------
# 3. McNemar tests between every pair of models (seed 42)
# ------------------------------------------------------------------
print("\n3. McNemar tests")
names = list(seed42_preds.keys())
aligned = {}
for name in names:
    pdf = seed42_preds[name].set_index("rid")
    aligned[name] = pdf
mc_rows = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        common = aligned[a].index.intersection(aligned[b].index)
        ca = (aligned[a].loc[common, "y"] == aligned[a].loc[common, "p"]).values
        cb = (aligned[b].loc[common, "y"] == aligned[b].loc[common, "p"]).values
        b_count = int(np.sum(ca & ~cb))
        c_count = int(np.sum(~ca & cb))
        if b_count + c_count == 0:
            stat, pval = 0.0, 1.0
        else:
            stat = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
            pval = math.erfc(math.sqrt(stat / 2))  # chi-square, 1 df
        mc_rows.append({"model_a": a, "model_b": b, "a_right_b_wrong": b_count,
                        "a_wrong_b_right": c_count, "chi2": round(stat, 3), "p_value": pval})
save(pd.DataFrame(mc_rows), "mcnemar_tests_seed42.csv")

# ------------------------------------------------------------------
# 4. Learning curve (seed 42)
# ------------------------------------------------------------------
print("\n4. Learning curve")
train_raw, test_raw = balanced.randomSplit([0.8, 0.2], seed=BASE_SEED)
lc_rows = []
for share in [0.1, 0.25, 0.5, 1.0]:
    sub = train_raw if share == 1.0 else train_raw.sample(fraction=share, seed=BASE_SEED)
    tr, te = prepare(sub, test_raw, numeric, cats, "standard")
    for name, est in [("Logistic Regression", LogisticRegression(maxIter=30, regParam=0.05, family="multinomial")),
                      ("Random Forest", RandomForestClassifier(numTrees=30, maxDepth=8, subsamplingRate=0.7,
                                                               featureSubsetStrategy="sqrt", seed=BASE_SEED))]:
        pdf = predictions_frame(est.fit(tr).transform(te))
        lc_rows.append({"train_share": share, "train_records": sub.count(), "model": name,
                        "accuracy": float(np.mean(pdf["y"].values == pdf["p"].values))})
        print(f"  share {share} {name}: {lc_rows[-1]['accuracy']:.4f}")
save(pd.DataFrame(lc_rows), "learning_curve.csv")

# ------------------------------------------------------------------
# 5. Feature-group ablation (seed 42)
# ------------------------------------------------------------------
print("\n5. Feature-group ablation")
ab_rows = []


def accuracy_for(num_cols, cat_cols, est):
    tr, te = prepare(train_raw, test_raw, num_cols, cat_cols, "standard")
    pdf = predictions_frame(est.fit(tr).transform(te))
    return float(np.mean(pdf["y"].values == pdf["p"].values))


def ablation_models():
    return {"Logistic Regression": LogisticRegression(maxIter=30, regParam=0.05, family="multinomial"),
            "Random Forest": RandomForestClassifier(numTrees=30, maxDepth=8, subsamplingRate=0.7,
                                                    featureSubsetStrategy="sqrt", seed=BASE_SEED)}


base = {}
for name, est in ablation_models().items():
    base[name] = accuracy_for(numeric, cats, est)
    ab_rows.append({"removed_group": "none (full model)", "model": name, "accuracy": base[name], "drop": 0.0})
for group, cols in GROUPS.items():
    num2 = [c for c in numeric if c not in cols]
    cat2 = [c for c in cats if c not in cols]
    if not num2 and not cat2:
        continue
    for name, est in ablation_models().items():
        acc = accuracy_for(num2, cat2, est)
        ab_rows.append({"removed_group": group, "model": name, "accuracy": acc, "drop": base[name] - acc})
        print(f"  without {group}, {name}: {acc:.4f}")
save(pd.DataFrame(ab_rows), "ablation_seed42.csv")

# ------------------------------------------------------------------
# 6. Label-derivation check (shallow trees, seed 42)
# ------------------------------------------------------------------
print("\n6. Label-derivation check")
lab_rows = []
checks = [("Features without Stress_Score", numeric, cats)]
if has_score:
    checks.append(("Stress_Score only", ["Stress_Score"], []))
    checks.append(("Features including Stress_Score", numeric_with_score, cats))
for label_text, num_cols, cat_cols in checks:
    for depth in [2, 5]:
        est = DecisionTreeClassifier(maxDepth=depth, seed=BASE_SEED)
        acc = accuracy_for(num_cols, cat_cols, est)
        lab_rows.append({"check": label_text, "tree_depth": depth, "accuracy": acc})
        print(f"  {label_text}, depth {depth}: {acc:.4f}")
save(pd.DataFrame(lab_rows), "label_derivation_check.csv")

run["total_seconds"] = round(time.time() - t_start, 1)
with open(os.path.join(args.out, "run_summary.json"), "w", encoding="utf-8") as f:
    json.dump(run, f, indent=2, default=str)
print("\nDone. Results in:", args.out)
spark.stop()
