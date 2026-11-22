# ============================================================
# STUDENT STRESS PREDICTION - BIG DATA ML + PLOTS (one file)
# Part A: trains and evaluates the models with PySpark, saves CSVs
# Part B: reads those CSVs and saves all plots to <OUTPUT_DIR>/plots
#
# Run in VS Code terminal:
#   python stress_ml_with_plots.py                      (train + plots)
#   python stress_ml_with_plots.py --plots-only         (only redraw plots)
#   python stress_ml_with_plots.py --label 15GB --data "D:\THESIS PROJECT\Stress_Data_15GB.csv" --out "D:\THESIS PROJECT\ML_15GB_results"
# ============================================================
#
# What changed compared with the previous version
#   1. Split FIRST, then balance ONLY the training data.
#      The test set keeps the real class distribution
#      (High ~86%, Medium ~14%, Low ~0.6%).
#   2. Labels are fixed as Low=0, Medium=1, High=2
#      (before, alphabetical order gave High=0, Low=1, Medium=2).
#   3. Feature-importance names are read from "raw_features"
#      (the scaler drops the names, which caused Unknown_Feature_x).
#   4. All metrics come from ONE confusion matrix per model:
#      accuracy, weighted + macro P/R/F1, per-class P/R/F1,
#      balanced accuracy, MCC and Cohen's Kappa.
#   5. Stage-by-stage timing is saved for the scalability study.
#   6. Results are saved as CSV and plotted automatically (Part B),
#      so no Spark/model variables are needed for plotting.
#   7. Naive Bayes inputs are clipped to [0, 1] (test values outside the
#      training range made MinMax output negative and crashed NB).
# ============================================================

import os
import sys
import csv
import json
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # save plots to files, no pop-up windows
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. CONFIGURATION  (edit these for 15 GB / 20 GB runs)
# ------------------------------------------------------------

DATASET_LABEL = "10GB"
DATA_PATH = r"D:\THESIS PROJECT\Stress_Data_10GB.csv"
OUTPUT_DIR = r"D:\THESIS PROJECT\ML_10GB_results"

JAVA_HOME = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"

DRIVER_MEMORY = "10g"
SHUFFLE_PARTITIONS = "16"

TRAIN_ROWS_PER_CLASS = 10000     # balanced training rows per class
TEST_EVAL_FRACTION = 0.10        # fraction of the 20% test split to evaluate
                                 # (natural distribution kept; 1.0 = full test split)
INCLUDE_RANDOM_FOREST = True     # your methodology + base paper use RF
TOP_K = 15                       # number of features shown in importance plots
SEED = 42

CLASS_ORDER = ["Low", "Medium", "High"]          # label 0, 1, 2
EXCLUDED_COLUMNS = ["Record_ID", "Stress_Score", "Stress_Level", "Comment", "Timestamp"]

# Optional command-line overrides:
#   python stress_ml_with_plots.py --label 15GB --data "D:\...\Stress_Data_15GB.csv" --out "D:\...\ML_15GB_results"
args = sys.argv[1:]
for flag, name in [("--label", "DATASET_LABEL"), ("--data", "DATA_PATH"), ("--out", "OUTPUT_DIR")]:
    if flag in args:
        globals()[name] = args[args.index(flag) + 1]

if os.path.isdir(JAVA_HOME):
    os.environ["JAVA_HOME"] = JAVA_HOME
    os.environ["PATH"] = os.path.join(JAVA_HOME, "bin") + os.pathsep + os.environ["PATH"]

# Spark workers must use the same Python as VS Code (avoids worker crashes on Windows)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

os.makedirs(OUTPUT_DIR, exist_ok=True)

from pyspark import StorageLevel  # noqa: E402
from pyspark.sql import SparkSession, functions as F  # noqa: E402
from pyspark.sql.types import NumericType, StringType  # noqa: E402
from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.feature import (  # noqa: E402
    Imputer, StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, MinMaxScaler
)
from pyspark.ml.functions import vector_to_array, array_to_vector  # noqa: E402
from pyspark.ml.classification import (  # noqa: E402
    LogisticRegression, DecisionTreeClassifier, NaiveBayes,
    MultilayerPerceptronClassifier, RandomForestClassifier
)


# ------------------------------------------------------------
# 2. HELPERS
# ------------------------------------------------------------

def out(name):
    return os.path.join(OUTPUT_DIR, f"{DATASET_LABEL}_{name}")


def save_csv(name, header, rows):
    path = out(name)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("Saved:", path)


stage_times = []


class Stage:
    """with Stage("name"): ...  -> records how long the block took"""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print(f"\n>>> {self.name}")
        self.t0 = time.time()

    def __exit__(self, *exc):
        sec = round(time.time() - self.t0, 2)
        stage_times.append([self.name, sec])
        print(f"<<< {self.name}: {sec} s")


def confusion_matrix(predictions, k=3):
    m = [[0] * k for _ in range(k)]
    for r in predictions.groupBy("label", "prediction").count().collect():
        m[int(r["label"])][int(r["prediction"])] += int(r["count"])
    return m


def metrics_from_cm(m):
    k = len(m)
    total = sum(sum(row) for row in m)
    support = [sum(m[i]) for i in range(k)]
    predicted = [sum(m[i][j] for i in range(k)) for j in range(k)]
    tp = [m[i][i] for i in range(k)]

    def div(a, b):
        return a / b if b else 0.0

    prec = [div(tp[i], predicted[i]) for i in range(k)]
    rec = [div(tp[i], support[i]) for i in range(k)]
    f1 = [div(2 * prec[i] * rec[i], prec[i] + rec[i]) for i in range(k)]
    w = [div(s, total) for s in support]
    acc = div(sum(tp), total)

    # Cohen's Kappa
    pe = sum(support[i] * predicted[i] for i in range(k)) / (total * total) if total else 0
    kappa = div(acc - pe, 1 - pe)
    # Multiclass Matthews Correlation Coefficient
    c, s = sum(tp), total
    num = c * s - sum(predicted[i] * support[i] for i in range(k))
    den = ((s * s - sum(p * p for p in predicted)) * (s * s - sum(t * t for t in support))) ** 0.5
    mcc = div(num, den)

    return {
        "Accuracy": acc,
        "Weighted_Precision": sum(p * x for p, x in zip(prec, w)),
        "Weighted_Recall": sum(r * x for r, x in zip(rec, w)),
        "Weighted_F1": sum(f * x for f, x in zip(f1, w)),
        "Macro_Precision": sum(prec) / k,
        "Macro_Recall": sum(rec) / k,
        "Macro_F1": sum(f1) / k,
        "Balanced_Accuracy": sum(rec) / k,
        "MCC": mcc,
        "Kappa": kappa,
        "per_class": [(CLASS_ORDER[i], prec[i], rec[i], f1[i], support[i]) for i in range(k)],
    }


def source_attribute(encoded_name, numeric_cols, categorical_cols):
    for c in numeric_cols:
        if encoded_name in (c, c + "_imputed"):
            return c
    for c in sorted(categorical_cols, key=len, reverse=True):
        if encoded_name.startswith(c + "_encoded"):
            return c
    return encoded_name


# ------------------------------------------------------------
# PLOT HELPERS
# ------------------------------------------------------------

CLASS_COLORS = {"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#e76f51"}


def read(name):
    p = out(name)
    if not os.path.isfile(p):
        print("Missing (skipped):", p)
        return None
    return pd.read_csv(p, encoding="utf-8-sig")


def save_plot(fig, name):
    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    p = os.path.join(plot_dir, f"{DATASET_LABEL}_{name}")
    fig.tight_layout()
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", p)


def pretty(s):
    return str(s).replace("_imputed", "").replace("_encoded_", " = ").replace("_", " ")


# ============================================================
# PART A - TRAINING AND EVALUATION
# ============================================================

def run_training():
    # ------------------------------------------------------------
    # 3. START SPARK
    # ------------------------------------------------------------

    total_start = time.time()

    with Stage("01_spark_start"):
        spark = (
            SparkSession.builder
            .appName(f"StudentStress_{DATASET_LABEL}_Models")
            .master("local[*]")
            .config("spark.driver.memory", DRIVER_MEMORY)
            .config("spark.sql.shuffle.partitions", SHUFFLE_PARTITIONS)
            .config("spark.sql.files.maxPartitionBytes", "128m")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")

    try:
        print("=" * 65)
        print(f"STUDENT STRESS PREDICTION - {DATASET_LABEL}")
        print("=" * 65)

        if not os.path.isfile(DATA_PATH):
            raise FileNotFoundError("Dataset not found: " + DATA_PATH)

        # --------------------------------------------------------
        # 4. LOAD
        # --------------------------------------------------------
        with Stage("02_load_schema"):
            df = (
                spark.read
                .option("header", "true")
                .option("inferSchema", "true")
                .option("samplingRatio", 0.01)   # infer types from 1% instead of reading all 10 GB twice
                .csv(DATA_PATH)
            )
            if "Stress_Level" not in df.columns:
                raise ValueError("Stress_Level column is missing.")

        # --------------------------------------------------------
        # 5. PROFILE (records, partitions, missing values, classes)
        # --------------------------------------------------------
        with Stage("03_profile"):
            agg = df.agg(
                F.count(F.lit(1)).alias("__rows"),
                *[F.sum(F.col(f"`{c}`").isNull().cast("int")).alias(c) for c in df.columns]
            ).collect()[0].asDict()
            total_records = agg.pop("__rows")
            total_attributes = len(df.columns)
            partitions = df.rdd.getNumPartitions()
            missing = {c: int(v or 0) for c, v in agg.items() if v}

            df = df.filter(F.col("Stress_Level").isin(*CLASS_ORDER))
            class_counts = {r["Stress_Level"]: int(r["count"])
                            for r in df.groupBy("Stress_Level").count().collect()}

        print(f"Total records: {total_records:,} | attributes: {total_attributes} | partitions: {partitions}")
        print("Missing values:", missing)
        print("Original class distribution:", class_counts)

        # --------------------------------------------------------
        # 6. FEATURES (target leakage removed)
        # --------------------------------------------------------
        feature_columns = [c for c in df.columns if c not in EXCLUDED_COLUMNS]
        numeric_columns = [f.name for f in df.schema.fields
                           if f.name in feature_columns and isinstance(f.dataType, NumericType)]
        categorical_columns = [f.name for f in df.schema.fields
                               if f.name in feature_columns and isinstance(f.dataType, StringType)]
        ignored_columns = [c for c in feature_columns if c not in numeric_columns + categorical_columns]
        print("Numeric predictors:", len(numeric_columns), "| Categorical predictors:", len(categorical_columns))
        if ignored_columns:
            print("Ignored unsupported columns:", ignored_columns)

        label_expr = (F.when(F.col("Stress_Level") == "Low", 0.0)
                       .when(F.col("Stress_Level") == "Medium", 1.0)
                       .otherwise(2.0))
        data = df.withColumn("label", label_expr)
        if categorical_columns:
            data = data.fillna("Unknown", subset=categorical_columns)
        data = data.select(numeric_columns + categorical_columns + ["label"])

        # --------------------------------------------------------
        # 7. SPLIT FIRST, THEN BALANCE TRAINING DATA ONLY
        # --------------------------------------------------------
        with Stage("04_split_and_balance"):
            train_all, test_all = data.randomSplit([0.8, 0.2], seed=SEED)

            fractions = {}
            for i, name in enumerate(CLASS_ORDER):
                n_train = class_counts.get(name, 0) * 0.8
                fractions[float(i)] = 1.0 if n_train == 0 else min(1.0, TRAIN_ROWS_PER_CLASS * 1.05 / n_train)
            train_df = train_all.sampleBy("label", fractions=fractions, seed=SEED).persist()

            test_df = test_all if TEST_EVAL_FRACTION >= 1.0 else test_all.sample(fraction=TEST_EVAL_FRACTION, seed=SEED)
            test_df = test_df.persist(StorageLevel.MEMORY_AND_DISK)

            train_dist = {CLASS_ORDER[int(r["label"])]: int(r["count"])
                          for r in train_df.groupBy("label").count().collect()}
            test_dist = {CLASS_ORDER[int(r["label"])]: int(r["count"])
                         for r in test_df.groupBy("label").count().collect()}
            train_count, test_count = sum(train_dist.values()), sum(test_dist.values())

        print("Balanced TRAINING distribution:", train_dist, f"(total {train_count:,})")
        print("Natural TEST distribution:   ", test_dist, f"(total {test_count:,})")

        # --------------------------------------------------------
        # 8. PREPROCESSING (fitted on training data only)
        # --------------------------------------------------------
        with Stage("05_preprocessing"):
            stages = []
            imputed_columns = [c + "_imputed" for c in numeric_columns]
            if numeric_columns:
                stages.append(Imputer(inputCols=numeric_columns, outputCols=imputed_columns, strategy="median"))
            encoded_columns = []
            if categorical_columns:
                indexed_columns = [c + "_indexed" for c in categorical_columns]
                encoded_columns = [c + "_encoded" for c in categorical_columns]
                stages.append(StringIndexer(inputCols=categorical_columns, outputCols=indexed_columns,
                                            handleInvalid="keep"))
                stages.append(OneHotEncoder(inputCols=indexed_columns, outputCols=encoded_columns,
                                            handleInvalid="keep"))
            stages.append(VectorAssembler(inputCols=imputed_columns + encoded_columns,
                                          outputCol="raw_features", handleInvalid="keep"))
            stages.append(StandardScaler(inputCol="raw_features", outputCol="features",
                                         withMean=False, withStd=True))

            preprocessing_model = Pipeline(stages=stages).fit(train_df)

            # Naive Bayes (multinomial) needs non-negative inputs -> MinMax on raw features
            train_prepared = preprocessing_model.transform(train_df)
            nb_scaler_model = MinMaxScaler(inputCol="raw_features", outputCol="nb_features").fit(train_prepared)

            # test values outside the training min/max would become negative -> clip to [0, 1]
            nb_clip = array_to_vector(F.transform(vector_to_array("nb_features"),
                                                  lambda x: F.least(F.greatest(x, F.lit(0.0)), F.lit(1.0))))

            keep = ["label", "raw_features", "features", "nb_features"]
            train_prepared = (nb_scaler_model.transform(train_prepared)
                              .withColumn("nb_features", nb_clip).select(*keep).persist())
            test_prepared = (nb_scaler_model.transform(preprocessing_model.transform(test_df))
                             .withColumn("nb_features", nb_clip)
                             .select(*keep).persist(StorageLevel.MEMORY_AND_DISK))
            train_prepared.count()
            test_prepared.count()

            # encoded feature names (from raw_features - the scaler output has no names)
            attrs = train_prepared.schema["raw_features"].metadata.get("ml_attr", {}).get("attrs", {})
            feature_names = {}
            for group in attrs.values():
                for a in group:
                    feature_names[int(a["idx"])] = a["name"]
            feature_count = len(train_prepared.first()["features"])

        print("Feature vector size:", feature_count)
        save_csv("class_label_mapping.csv", ["Encoded_Label", "Stress_Level"],
                 [[i, c] for i, c in enumerate(CLASS_ORDER)])

        # --------------------------------------------------------
        # 9. MODELS
        # --------------------------------------------------------
        models = [
            ("Logistic Regression",
             LogisticRegression(featuresCol="features", labelCol="label",
                                maxIter=30, regParam=0.05, family="multinomial"), "features"),
            ("Decision Tree",
             DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=8, seed=SEED), "features"),
            ("Naive Bayes",
             NaiveBayes(featuresCol="nb_features", labelCol="label",
                        smoothing=1.0, modelType="multinomial"), "nb_features"),
            ("Multilayer Perceptron",
             MultilayerPerceptronClassifier(featuresCol="features", labelCol="label",
                                            layers=[feature_count, 64, 32, 3], maxIter=100,
                                            blockSize=128, seed=SEED), "features"),
        ]
        if INCLUDE_RANDOM_FOREST:
            models.append(
                ("Random Forest",
                 RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100,
                                        maxDepth=10, featureSubsetStrategy="sqrt", seed=SEED), "features"))

        # --------------------------------------------------------
        # 10. TRAIN + EVALUATE
        # --------------------------------------------------------
        comparison_rows, per_class_rows, confusion_rows, importance_rows = [], [], [], []

        for model_name, classifier, _ in models:
            print("\n" + "=" * 65 + f"\nTRAINING: {model_name}\n" + "=" * 65)

            t0 = time.time()
            fitted = classifier.fit(train_prepared)
            train_sec = time.time() - t0

            t0 = time.time()
            cm = confusion_matrix(fitted.transform(test_prepared))
            predict_sec = time.time() - t0
            stage_times.append([f"06_train_{model_name}", round(train_sec, 2)])
            stage_times.append([f"07_evaluate_{model_name}", round(predict_sec, 2)])

            m = metrics_from_cm(cm)
            print(f"Accuracy {m['Accuracy']:.4f} | Macro-F1 {m['Macro_F1']:.4f} | "
                  f"Weighted-F1 {m['Weighted_F1']:.4f} | MCC {m['MCC']:.4f} | "
                  f"train {train_sec:.1f}s, evaluate {predict_sec:.1f}s")
            for cls, p, r, f, s in m["per_class"]:
                print(f"   {cls:<7} precision {p:.3f}  recall {r:.3f}  F1 {f:.3f}  (n={s:,})")

            comparison_rows.append([model_name, m["Accuracy"], m["Weighted_Precision"], m["Weighted_Recall"],
                                    m["Weighted_F1"], m["Macro_Precision"], m["Macro_Recall"], m["Macro_F1"],
                                    m["Balanced_Accuracy"], m["MCC"], m["Kappa"], train_sec, predict_sec])
            for cls, p, r, f, s in m["per_class"]:
                per_class_rows.append([model_name, cls, p, r, f, s])
            for i in range(3):
                for j in range(3):
                    confusion_rows.append([model_name, i, CLASS_ORDER[i], j, CLASS_ORDER[j], cm[i][j]])

            # feature importance for tree models
            if hasattr(fitted, "featureImportances"):
                imp = fitted.featureImportances.toArray()
                for idx, val in enumerate(imp):
                    name = feature_names.get(idx, f"Unknown_Feature_{idx}")
                    importance_rows.append([model_name, idx, name,
                                            source_attribute(name, numeric_columns, categorical_columns),
                                            float(val)])
            # coefficients for logistic regression
            if model_name == "Logistic Regression":
                W = fitted.coefficientMatrix.toArray()
                save_csv("logistic_regression_coefficients.csv",
                         ["Feature_Index", "Feature_Name", "Attribute"] + [f"Coef_{c}" for c in CLASS_ORDER],
                         [[idx, feature_names.get(idx, f"f{idx}"),
                           source_attribute(feature_names.get(idx, f"f{idx}"), numeric_columns, categorical_columns)]
                          + [float(W[k][idx]) for k in range(3)] for idx in range(W.shape[1])])

        # --------------------------------------------------------
        # 11. SAVE RESULTS
        # --------------------------------------------------------
        save_csv("model_comparison.csv",
                 ["Model", "Accuracy", "Weighted_Precision", "Weighted_Recall", "Weighted_F1",
                  "Macro_Precision", "Macro_Recall", "Macro_F1", "Balanced_Accuracy", "MCC", "Kappa",
                  "Train_Seconds", "Evaluate_Seconds"], comparison_rows)
        save_csv("per_class_metrics.csv", ["Model", "Stress_Level", "Precision", "Recall", "F1", "Support"],
                 per_class_rows)
        save_csv("confusion_matrices.csv",
                 ["Model", "Actual_Label", "Actual_Stress_Level", "Predicted_Label", "Predicted_Stress_Level", "Count"],
                 confusion_rows)
        save_csv("feature_importance.csv", ["Model", "Feature_Index", "Feature_Name", "Attribute", "Importance"],
                 sorted(importance_rows, key=lambda r: (r[0], -r[4])))

        total_elapsed = time.time() - total_start
        stage_times.append(["total", round(total_elapsed, 2)])
        save_csv("stage_times.csv", ["Stage", "Seconds"], stage_times)

        summary = {
            "dataset": DATASET_LABEL, "data_path": DATA_PATH,
            "file_size_gb": round(os.path.getsize(DATA_PATH) / 1024 ** 3, 3),
            "records": total_records, "attributes": total_attributes, "partitions": partitions,
            "cores": spark.sparkContext.defaultParallelism, "spark_version": spark.version,
            "driver_memory": DRIVER_MEMORY, "missing_values": missing,
            "class_distribution": class_counts,
            "train_distribution_balanced": train_dist, "test_distribution_natural": test_dist,
            "test_eval_fraction": TEST_EVAL_FRACTION, "feature_vector_size": feature_count,
            "numeric_predictors": numeric_columns, "categorical_predictors": categorical_columns,
            "excluded_columns": EXCLUDED_COLUMNS, "stage_seconds": dict(stage_times),
        }
        with open(out("run_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("Saved:", out("run_summary.json"))

        print("\n" + "=" * 65 + f"\n{DATASET_LABEL} MACHINE LEARNING COMPLETED\n" + "=" * 65)
        print(f"Records: {total_records:,} | train: {train_count:,} (balanced) | test: {test_count:,} (natural)")
        print(f"Total elapsed: {total_elapsed:.1f} s")

        for d in (train_df, test_df, train_prepared, test_prepared):
            d.unpersist()

    finally:
        spark.stop()


# ============================================================
# PART B - PLOTS (reads the saved CSV files)
# ============================================================

def make_plots():
    print("\n" + "=" * 65 + f"\nCREATING PLOTS - {DATASET_LABEL}\n" + "=" * 65)
    comparison = read("model_comparison.csv")
    per_class = read("per_class_metrics.csv")
    confusion = read("confusion_matrices.csv")
    importance = read("feature_importance.csv")
    coefs = read("logistic_regression_coefficients.csv")
    stages = read("stage_times.csv")
    summary = None
    if os.path.isfile(out("run_summary.json")):
        with open(out("run_summary.json"), encoding="utf-8") as f:
            summary = json.load(f)

    # ------------------------------------------------------------
    # 1. Overall model comparison
    # ------------------------------------------------------------
    if comparison is not None:
        metrics = ["Accuracy", "Macro_F1", "Weighted_F1", "Balanced_Accuracy", "MCC"]
        metrics = [m for m in metrics if m in comparison.columns]
        x = np.arange(len(comparison))
        w = 0.8 / len(metrics)
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for i, m in enumerate(metrics):
            bars = ax.bar(x + (i - (len(metrics) - 1) / 2) * w, comparison[m], w, label=m.replace("_", " "))
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}",
                        ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_xticks(x, comparison["Model"], rotation=10)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("score")
        ax.set_title(f"Model comparison - {DATASET_LABEL} (test set with natural class distribution)")
        ax.legend(ncol=len(metrics), fontsize=8, loc="upper center")
        save_plot(fig, "model_comparison.png")

        print("\n" + comparison[["Model"] + metrics].round(4).to_string(index=False))

    # ------------------------------------------------------------
    # 2. Per-class recall and F1 (shows minority-class performance)
    # ------------------------------------------------------------
    if per_class is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        models = per_class["Model"].unique()
        x = np.arange(len(models))
        w = 0.26
        for ax, metric in zip(axes, ["Recall", "F1"]):
            for i, cls in enumerate(CLASS_ORDER):
                vals = [per_class[(per_class.Model == m) & (per_class.Stress_Level == cls)][metric].values[0]
                        for m in models]
                ax.bar(x + (i - 1) * w, vals, w, label=cls, color=CLASS_COLORS[cls])
            ax.set_xticks(x, models, rotation=15)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"Per-class {metric}")
        axes[0].set_ylabel("score")
        axes[1].legend(title="Stress level")
        fig.suptitle(f"Per-class performance - {DATASET_LABEL}")
        save_plot(fig, "per_class_recall_f1.png")

    # ------------------------------------------------------------
    # 3. Confusion matrices (labelled, counts + row %)
    # ------------------------------------------------------------
    if confusion is not None:
        models = confusion["Model"].unique()
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.4))
        axes = np.atleast_1d(axes)
        for ax, model in zip(axes, models):
            sub = confusion[confusion.Model == model]
            M = np.zeros((3, 3))
            for _, r in sub.iterrows():
                M[int(r.Actual_Label), int(r.Predicted_Label)] = r.Count
            norm = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{int(M[i, j]):,}\n({norm[i, j]:.1%})", ha="center", va="center",
                            fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
            ax.set_xticks(range(3), CLASS_ORDER)
            ax.set_yticks(range(3), CLASS_ORDER)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(model, fontsize=10)
        fig.suptitle(f"Confusion matrices - {DATASET_LABEL} (row % = recall)")
        save_plot(fig, "confusion_matrices.png")

    # ------------------------------------------------------------
    # 4. Feature importance (tree models), grouped to original attributes
    # ------------------------------------------------------------
    if importance is not None and len(importance):
        for model in importance["Model"].unique():
            sub = importance[importance.Model == model]
            agg = sub.groupby("Attribute")["Importance"].sum().sort_values(ascending=False).head(TOP_K)[::-1]
            fig, ax = plt.subplots(figsize=(8, 0.42 * len(agg) + 1.5))
            ax.barh([pretty(a) for a in agg.index], agg.values, color="#264653")
            ax.set_xlabel("importance (one-hot columns summed per attribute)")
            ax.set_title(f"{model} - top {TOP_K} stress factors ({DATASET_LABEL})")
            save_plot(fig, f"feature_importance_{model.replace(' ', '_')}.png")
            agg[::-1].to_csv(os.path.join(OUTPUT_DIR, "plots", f"{DATASET_LABEL}_feature_importance_{model.replace(' ', '_')}_by_attribute.csv"))

    # ------------------------------------------------------------
    # 5. Logistic Regression coefficients (direction of effect per class)
    # ------------------------------------------------------------
    if coefs is not None:
        cols = [f"Coef_{c}" for c in CLASS_ORDER]
        c = coefs.copy()
        c["strength"] = c[cols].abs().max(axis=1)
        c = c.sort_values("strength", ascending=False).head(TOP_K)
        M = c[cols].values
        lim = np.abs(M).max() or 1
        fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(c) + 1.5))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(3), CLASS_ORDER)
        ax.set_yticks(range(len(c)), [pretty(n) for n in c.Feature_Name])
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="coefficient (standardised features)")
        ax.set_title(f"Logistic Regression coefficients - {DATASET_LABEL}")
        save_plot(fig, "logistic_regression_coefficients.png")

    # ------------------------------------------------------------
    # 6. Runtime per model and per stage
    # ------------------------------------------------------------
    if comparison is not None and "Train_Seconds" in comparison:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.barh(comparison.Model, comparison.Train_Seconds, label="training", color="#264653")
        ax.barh(comparison.Model, comparison.Evaluate_Seconds, left=comparison.Train_Seconds,
                label="prediction + evaluation", color="#2a9d8f")
        ax.set_xlabel("seconds")
        ax.set_title(f"Model runtime - {DATASET_LABEL}")
        ax.legend()
        save_plot(fig, "model_runtime.png")

    if stages is not None:
        s = stages[stages.Stage != "total"]
        fig, ax = plt.subplots(figsize=(9, 0.38 * len(s) + 1.5))
        ax.barh(s.Stage[::-1], s.Seconds[::-1], color="#e76f51")
        total = stages[stages.Stage == "total"].Seconds
        ax.set_xlabel("seconds")
        ax.set_title(f"Pipeline stage times - {DATASET_LABEL}"
                     + (f" (total {total.values[0]:.0f} s)" if len(total) else ""))
        save_plot(fig, "stage_times.png")

    # ------------------------------------------------------------
    # 7. Class distribution: original vs balanced training vs test
    # ------------------------------------------------------------
    if summary:
        dists = [("Original data", summary["class_distribution"]),
                 ("Training (balanced)", summary["train_distribution_balanced"]),
                 ("Test (natural)", summary["test_distribution_natural"])]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (title, d) in zip(axes, dists):
            vals = [d.get(c, 0) for c in CLASS_ORDER]
            tot = sum(vals) or 1
            bars = ax.bar(CLASS_ORDER, vals, color=[CLASS_COLORS[c] for c in CLASS_ORDER])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:,}\n{v / tot:.1%}",
                        ha="center", va="bottom", fontsize=8)
            ax.set_title(title)
            ax.set_ylim(0, max(vals) * 1.25 if max(vals) else 1)
        fig.suptitle(f"Stress-level distribution - {DATASET_LABEL}")
        save_plot(fig, "class_distribution.png")

    print("\nAll plots saved in:", os.path.join(OUTPUT_DIR, "plots"))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if "--plots-only" not in sys.argv:
        run_training()
    make_plots()