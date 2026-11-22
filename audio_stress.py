"""
Speaker-level stress classification from audio features (PySpark).

Input : features CSV from extract_audio_features.py
        (columns: file, speaker_id, label, <numeric features>)
Output: per-repeat metrics, summary, and summed confusion matrices

Methodology (mirrors multi_dataset_stress.py, adapted for audio):
  - Split by SPEAKER, not by clip, so no student appears in both train and test.
  - Binary label: positive class vs everything else.
  - Training set balanced by exact sampling; test set keeps its natural distribution.
  - Five repeats, each with a new speaker split and new seed.
  - Five models: Logistic Regression, Decision Tree, Naive Bayes, MLP, Random Forest.
  - Metrics from the confusion matrix: accuracy, macro-F1, MCC, balanced accuracy,
    per-class recall, Cohen's kappa.

Usage:
  python audio_stress_pipeline.py --features audio_features.csv --positive Stressed --out results
"""
import argparse
import os

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    LogisticRegression,
    MultilayerPerceptronClassifier,
    NaiveBayes,
    RandomForestClassifier,
)
from pyspark.ml.feature import Imputer, MinMaxScaler, StandardScaler, VectorAssembler
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

REPEATS = 5
TEST_FRACTION = 0.2
BASE_SEED = 42
NON_FEATURE_COLS = {"file", "speaker_id", "label", "label_bin"}


def build_model(name, n_features, seed):
    if name == "Logistic Regression":
        return LogisticRegression(maxIter=100, regParam=0.01, featuresCol="features", labelCol="label")
    if name == "Decision Tree":
        return DecisionTreeClassifier(maxDepth=6, featuresCol="features", labelCol="label", seed=seed)
    if name == "Naive Bayes":
        return NaiveBayes(modelType="multinomial", smoothing=1.0, featuresCol="features", labelCol="label")
    if name == "Multilayer Perceptron":
        return MultilayerPerceptronClassifier(
            layers=[n_features, 32, 16, 2], maxIter=1000, blockSize=64, seed=seed,
            featuresCol="features", labelCol="label")
    if name == "Random Forest":
        return RandomForestClassifier(numTrees=100, maxDepth=8, featureSubsetStrategy="sqrt",
                                      featuresCol="features", labelCol="label", seed=seed)
    raise ValueError(name)


MODELS = ["Logistic Regression", "Decision Tree", "Naive Bayes",
          "Multilayer Perceptron", "Random Forest"]


def speaker_split(df, seed):
    """Assign whole speakers to train or test."""
    speakers = sorted(df.select("speaker_id").distinct().toPandas()["speaker_id"].tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)
    n_test = max(1, round(TEST_FRACTION * len(speakers)))
    test_speakers = speakers[:n_test]
    return df.withColumn("is_test", F.col("speaker_id").isin(test_speakers)), n_test, len(speakers)


def balance_train(train, seed):
    """Sample each class down to the size of the smallest class."""
    counts = {r["label"]: r["count"] for r in train.groupBy("label").count().collect()}
    n_min = min(counts.values())
    w = Window.partitionBy("label").orderBy(F.rand(seed))
    return (train.withColumn("rn", F.row_number().over(w))
                 .filter(F.col("rn") <= n_min).drop("rn")), n_min


def confusion(pred):
    """Return tn, fp, fn, tp for positive class = 1."""
    rows = pred.groupBy("label", "prediction").count().collect()
    cm = {(int(r["label"]), int(r["prediction"])): int(r["count"]) for r in rows}
    tn = cm.get((0, 0), 0)
    fp = cm.get((0, 1), 0)
    fn = cm.get((1, 0), 0)
    tp = cm.get((1, 1), 0)
    return tn, fp, fn, tp


def metrics_from_cm(tn, fp, fn, tp):
    n = tn + fp + fn + tp
    acc = (tp + tn) / n if n else float("nan")
    rec_pos = tp / (tp + fn) if (tp + fn) else float("nan")
    rec_neg = tn / (tn + fp) if (tn + fp) else float("nan")
    prec_pos = tp / (tp + fp) if (tp + fp) else 0.0
    prec_neg = tn / (tn + fn) if (tn + fn) else 0.0

    def f1(p, r):
        return 2 * p * r / (p + r) if (p + r) else 0.0

    macro_f1 = (f1(prec_pos, rec_pos) + f1(prec_neg, rec_neg)) / 2
    bal_acc = (rec_pos + rec_neg) / 2
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n) if n else 0.0
    kappa = (acc - pe) / (1 - pe) if (1 - pe) else 0.0
    return {"accuracy": acc, "macro_f1": macro_f1, "mcc": mcc, "balanced_acc": bal_acc,
            "recall_negative": rec_neg, "recall_positive": rec_pos, "kappa": kappa,
            "n_test": n}


def clip_to_train_range(train, test, cols):
    """Clip train and test values to the training min/max, so MinMax scaling stays in [0, 1].
    Needed for Naive Bayes, which rejects negative values."""
    bounds = train.agg(*[F.min(c).alias(f"{c}__lo") for c in cols],
                       *[F.max(c).alias(f"{c}__hi") for c in cols]).collect()[0]

    def clip(df):
        for c in cols:
            lo, hi = bounds[f"{c}__lo"], bounds[f"{c}__hi"]
            df = df.withColumn(c, F.least(F.greatest(F.col(c), F.lit(lo)), F.lit(hi)))
        return df

    return clip(train), clip(test)


def run_model(name, train, test, num_cols, seed):
    if name == "Naive Bayes":
        train, test = clip_to_train_range(train, test, num_cols)
    imputed = [f"{c}_imp" for c in num_cols]
    imputer = Imputer(inputCols=num_cols, outputCols=imputed, strategy="median")
    assembler = VectorAssembler(inputCols=imputed, outputCol="raw_features")
    if name == "Naive Bayes":
        scaler = MinMaxScaler(inputCol="raw_features", outputCol="features", min=0.0, max=1.0)
    else:
        scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=False)
    model = build_model(name, len(num_cols), seed)
    pipe = Pipeline(stages=[imputer, assembler, scaler, model])
    if name == "Multilayer Perceptron":
        train = train.coalesce(1)  # single partition: makes the MLP reproducible across core counts
    fitted = pipe.fit(train)
    pred = fitted.transform(test).select("label", "prediction")
    return confusion(pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--positive", required=True, help="label value treated as the positive class")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--driver-memory", default="4g")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    spark = (SparkSession.builder.master("local[*]").appName("audio_stress")
             .config("spark.driver.memory", args.driver_memory).getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    raw = pd.read_csv(args.features)
    raw["speaker_id"] = raw["speaker_id"].astype(str)
    raw["label_bin"] = (raw["label"].astype(str) == str(args.positive)).astype(int)
    num_cols = [c for c in raw.columns if c not in NON_FEATURE_COLS]
    raw = raw.dropna(axis=1, how="all")  # drop columns that are entirely empty
    num_cols = [c for c in num_cols if c in raw.columns]
    df = spark.createDataFrame(raw[["file", "speaker_id", "label_bin"] + num_cols]
                               .rename(columns={"label_bin": "label"}))
    df = df.select("file", "speaker_id", "label", *[F.col(c).cast("double") for c in num_cols])

    print(f"clips={df.count()} speakers={raw['speaker_id'].nunique()} features={len(num_cols)}")
    print(f"positive class = '{args.positive}' -> label 1")

    rows = []
    cm_sum = {m: np.zeros((2, 2), dtype=int) for m in MODELS}
    for rep in range(args.repeats):
        seed = BASE_SEED + rep
        split_df, n_test_spk, n_spk = speaker_split(df, seed)
        train_raw = split_df.filter(~F.col("is_test"))
        test = split_df.filter(F.col("is_test"))
        # speaker check: no speaker in both sets
        tr_spk = set(r["speaker_id"] for r in train_raw.select("speaker_id").distinct().collect())
        te_spk = set(r["speaker_id"] for r in test.select("speaker_id").distinct().collect())
        assert not (tr_spk & te_spk), "speaker leakage between train and test"
        train, n_min = balance_train(train_raw, seed)
        print(f"repeat {rep + 1}/{args.repeats}: train speakers={len(tr_spk)} test speakers={len(te_spk)} "
              f"balanced train per class={n_min}")
        for m in MODELS:
            tn, fp, fn, tp = run_model(m, train, test, num_cols, seed)
            cm_sum[m] += np.array([[tn, fp], [fn, tp]])
            met = metrics_from_cm(tn, fp, fn, tp)
            print(f"  {m:24s} acc={met['accuracy']:.3f} macroF1={met['macro_f1']:.3f} mcc={met['mcc']:.3f}")
            rows.append({"repeat": rep + 1, "model": m, **met})

    per_repeat = pd.DataFrame(rows)
    per_repeat.to_csv(os.path.join(args.out, "per_repeat_metrics.csv"), index=False)
    summary = (per_repeat.groupby("model")[["accuracy", "macro_f1", "mcc", "balanced_acc", "kappa"]]
               .agg(["mean", "std"]))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reindex(MODELS).reset_index()
    summary.to_csv(os.path.join(args.out, "summary_metrics.csv"), index=False)
    for m, cm in cm_sum.items():
        pd.DataFrame(cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]).to_csv(
            os.path.join(args.out, f"confusion_{m.replace(' ', '_').lower()}.csv"))

    print("\n=== Summary (mean over repeats) ===")
    print(summary[["model", "accuracy_mean", "macro_f1_mean", "mcc_mean", "kappa_mean"]]
          .round(3).to_string(index=False))
    print(f"\nSaved to {args.out}")
    spark.stop()


if __name__ == "__main__":
    main()