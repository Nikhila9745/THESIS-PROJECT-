# ================================================================
# STUDENT STRESS THESIS - RESULTS GENERATION
# ================================================================

import os
import sys

# ------------------------------------------------
# JAVA
# ------------------------------------------------

JAVA_HOME = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"

os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = (
    os.path.join(JAVA_HOME, "bin")
    + os.pathsep
    + os.environ["PATH"]
)

# ------------------------------------------------
# PYSPARK
# ------------------------------------------------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler,
    Imputer
)
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.storagelevel import StorageLevel

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ================================================================
# 1. SETTINGS
# ================================================================

DATASET_PATH = r"D:\THESIS PROJECT\Stress_Data_5GB.csv"

RESULTS_DIR = r"C:\Users\JP\THESIS PROJECT\results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

print("=" * 70)
print("STUDENT STRESS - RESULTS GENERATION")
print("=" * 70)

print("Results folder:")
print(RESULTS_DIR)


# ================================================================
# 2. SPARK SESSION
# ================================================================

spark = (
    SparkSession.builder
    .appName("StudentStress_Results")
    .master("local[*]")
    .config("spark.driver.memory", "10g")
    .config("spark.executor.memory", "10g")
    .config("spark.sql.shuffle.partitions", "16")
    .config("spark.sql.files.maxPartitionBytes", "128m")
    .config("spark.ui.showConsoleProgress", "true")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("\nSpark version:", spark.version)


# ================================================================
# 3. LOAD DATASET
# ================================================================

print("\nLoading 5 GB dataset...")

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(DATASET_PATH)
)

total_records = df.count()

print(
    "Total records:",
    f"{total_records:,}"
)


# ================================================================
# 4. STRESS DISTRIBUTION
# ================================================================

print("\nCalculating stress distribution...")

distribution = (
    df.groupBy("Stress_Level")
    .count()
    .withColumn(
        "Percentage",
        F.round(
            F.col("count") * 100 / total_records,
            2
        )
    )
)

distribution_pd = (
    distribution
    .orderBy(F.desc("count"))
    .toPandas()
)

distribution_pd.to_csv(
    os.path.join(
        RESULTS_DIR,
        "stress_distribution.csv"
    ),
    index=False
)

print(distribution_pd)


# ================================================================
# 5. FEATURES
# ================================================================

numeric_columns = [
    "Age",
    "Study_Hours",
    "Working_Hours",
    "Academic_Workload",
    "Exam_Pressure",
    "Assignment_Pressure",
    "Attendance_Percentage",
    "Sleep_Hours",
    "Screen_Time_Hours",
    "Social_Media_Hours",
    "Exercise_Minutes",
    "Water_Intake_Liters",
    "Financial_Stress",
    "Family_Stress",
    "Relationship_Stress",
    "Social_Support",
    "Anxiety_Level",
    "Mood_Score",
    "Heart_Rate",
    "Systolic_BP",
    "Diastolic_BP",
    "HRV",
    "Respiration_Rate",
    "Body_Temperature",
    "BMI",
    "Noise_Level",
    "Air_Quality_Index"
]

categorical_columns = [
    "Gender",
    "Education",
    "Occupation",
    "Marital_Status",
    "Diet_Quality",
    "Smoking",
    "Alcohol",
    "Weather",
    "Activity"
]


# ================================================================
# 6. REMOVE LEAKAGE / NON-PREDICTIVE COLUMNS
# ================================================================

df_model = df.drop(
    "Record_ID",
    "Stress_Score",
    "Comment",
    "Timestamp"
)


# ================================================================
# 7. CATEGORICAL CLEANING
# ================================================================

for c in categorical_columns:

    if c in df_model.columns:

        df_model = df_model.withColumn(
            c,
            F.coalesce(
                F.col(c).cast("string"),
                F.lit("Unknown")
            )
        )


# ================================================================
# 8. IMPUTATION
# ================================================================

existing_numeric = [
    c for c in numeric_columns
    if c in df_model.columns
]

imputer = Imputer(
    inputCols=existing_numeric,
    outputCols=[
        c + "_imputed"
        for c in existing_numeric
    ],
    strategy="median"
)


# ================================================================
# 9. TARGET ENCODING
# ================================================================

label_indexer = StringIndexer(
    inputCol="Stress_Level",
    outputCol="label",
    handleInvalid="keep"
)


# ================================================================
# 10. CATEGORICAL ENCODING
# ================================================================

existing_categorical = [
    c for c in categorical_columns
    if c in df_model.columns
]

indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_index",
        handleInvalid="keep"
    )
    for c in existing_categorical
]

encoders = [
    OneHotEncoder(
        inputCol=c + "_index",
        outputCol=c + "_encoded"
    )
    for c in existing_categorical
]


# ================================================================
# 11. FEATURE VECTOR
# ================================================================

imputed_numeric = [
    c + "_imputed"
    for c in existing_numeric
]

encoded_categorical = [
    c + "_encoded"
    for c in existing_categorical
]

feature_columns = (
    imputed_numeric +
    encoded_categorical
)

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features_raw",
    handleInvalid="keep"
)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withMean=False,
    withStd=True
)


# ================================================================
# 12. PREPROCESSING PIPELINE
# ================================================================

pipeline = Pipeline(
    stages=[
        imputer,
        label_indexer,
        *indexers,
        *encoders,
        assembler,
        scaler
    ]
)

print("\nRunning preprocessing...")

preprocessing_model = pipeline.fit(
    df_model
)

processed_df = (
    preprocessing_model
    .transform(df_model)
    .select(
        "Stress_Level",
        "label",
        "features"
    )
)


# ================================================================
# 13. CONTROLLED BALANCED SAMPLE
# ================================================================

print("\nCreating balanced sample...")

label_counts = {
    row["label"]: row["count"]
    for row in (
        processed_df
        .groupBy("label")
        .count()
        .collect()
    )
}

TARGET_PER_CLASS = min(
    10000,
    min(label_counts.values())
)

sample_parts = []

for label_value in sorted(
    label_counts.keys()
):

    class_df = processed_df.filter(
        F.col("label") == float(label_value)
    )

    class_count = label_counts[label_value]

    if class_count > TARGET_PER_CLASS:

        fraction = (
            TARGET_PER_CLASS /
            float(class_count)
        )

        sampled = (
            class_df
            .sample(
                False,
                fraction,
                seed=42
            )
            .limit(TARGET_PER_CLASS)
        )

    else:

        sampled = class_df

    sample_parts.append(sampled)


balanced_df = sample_parts[0]

for part in sample_parts[1:]:

    balanced_df = balanced_df.unionByName(
        part
    )


balanced_df = (
    balanced_df
    .orderBy(F.rand(seed=42))
)


balanced_df = balanced_df.persist(
    StorageLevel.DISK_ONLY
)

balanced_count = balanced_df.count()

print(
    "Balanced sample:",
    f"{balanced_count:,}"
)


# ================================================================
# 14. TRAIN / TEST
# ================================================================

train_df, test_df = balanced_df.randomSplit(
    [0.8, 0.2],
    seed=42
)

train_df = train_df.persist(
    StorageLevel.DISK_ONLY
)

test_df = test_df.persist(
    StorageLevel.DISK_ONLY
)

print(
    "Training:",
    f"{train_df.count():,}"
)

print(
    "Testing:",
    f"{test_df.count():,}"
)


# ================================================================
# 15. EVALUATION FUNCTION
# ================================================================

def calculate_metrics(
    model_name,
    predictions
):

    evaluator_accuracy = (
        MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
    )

    evaluator_precision = (
        MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedPrecision"
        )
    )

    evaluator_recall = (
        MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedRecall"
        )
    )

    evaluator_f1 = (
        MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1"
        )
    )

    accuracy = evaluator_accuracy.evaluate(
        predictions
    )

    precision = evaluator_precision.evaluate(
        predictions
    )

    recall = evaluator_recall.evaluate(
        predictions
    )

    f1 = evaluator_f1.evaluate(
        predictions
    )

    print("\n", model_name)

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    }


# ================================================================
# 16. LOGISTIC REGRESSION
# ================================================================

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION")
print("=" * 70)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=30,
    regParam=0.05,
    elasticNetParam=0.0,
    family="multinomial"
)

lr_model = lr.fit(
    train_df
)

lr_predictions = lr_model.transform(
    test_df
)

lr_results = calculate_metrics(
    "Logistic Regression",
    lr_predictions
)


# ================================================================
# 17. LR CONFUSION MATRIX
# ================================================================

lr_cm = (
    lr_predictions
    .groupBy(
        "label",
        "prediction"
    )
    .count()
    .orderBy(
        "label",
        "prediction"
    )
    .toPandas()
)

lr_cm.to_csv(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix_lr.csv"
    ),
    index=False
)


# ================================================================
# 18. FREE MEMORY
# ================================================================

lr_predictions.unpersist()
spark.catalog.clearCache()


# ================================================================
# 19. RANDOM FOREST
# ================================================================

print("\n" + "=" * 70)
print("RANDOM FOREST")
print("=" * 70)

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=30,
    maxDepth=8,
    featureSubsetStrategy="sqrt",
    subsamplingRate=0.7,
    seed=42
)

rf_model = rf.fit(
    train_df
)

rf_predictions = rf_model.transform(
    test_df
)

rf_results = calculate_metrics(
    "Random Forest",
    rf_predictions
)


# ================================================================
# 20. RF CONFUSION MATRIX
# ================================================================

rf_cm = (
    rf_predictions
    .groupBy(
        "label",
        "prediction"
    )
    .count()
    .orderBy(
        "label",
        "prediction"
    )
    .toPandas()
)

rf_cm.to_csv(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix_rf.csv"
    ),
    index=False
)


# ================================================================
# 21. MODEL COMPARISON
# ================================================================

results = pd.DataFrame([
    lr_results,
    rf_results
])

results.to_csv(
    os.path.join(
        RESULTS_DIR,
        "model_comparison.csv"
    ),
    index=False
)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results)


# ================================================================
# 22. FEATURE IMPORTANCE
# ================================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance_values = (
    rf_model
    .featureImportances
    .toArray()
)

importance_df = pd.DataFrame({
    "Feature_Index": np.arange(
        len(importance_values)
    ),
    "Importance": importance_values
})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
)

importance_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "feature_importance.csv"
    ),
    index=False
)

print(
    importance_df.head(20)
)


# ================================================================
# 23. MODEL COMPARISON GRAPH
# ================================================================

plt.figure(
    figsize=(10, 6)
)

x = np.arange(
    len(results["Model"])
)

width = 0.2

plt.bar(
    x - 1.5 * width,
    results["Accuracy"],
    width,
    label="Accuracy"
)

plt.bar(
    x - 0.5 * width,
    results["Precision"],
    width,
    label="Precision"
)

plt.bar(
    x + 0.5 * width,
    results["Recall"],
    width,
    label="Recall"
)

plt.bar(
    x + 1.5 * width,
    results["F1_Score"],
    width,
    label="F1 Score"
)

plt.xticks(
    x,
    results["Model"]
)

plt.ylim(
    0,
    1
)

plt.ylabel(
    "Score"
)

plt.title(
    "Student Stress Prediction - Model Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "model_comparison.png"
    ),
    dpi=300
)

plt.close()


# ================================================================
# 24. STRESS DISTRIBUTION GRAPH
# ================================================================

plt.figure(
    figsize=(8, 6)
)

plt.bar(
    distribution_pd["Stress_Level"],
    distribution_pd["Percentage"]
)

plt.ylabel(
    "Percentage (%)"
)

plt.xlabel(
    "Stress Level"
)

plt.title(
    "Student Stress Level Distribution"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "stress_distribution.png"
    ),
    dpi=300
)

plt.close()


# ================================================================
# 25. CONFUSION MATRIX FUNCTION
# ================================================================

def create_confusion_plot(
    cm_df,
    filename,
    title
):

    matrix = (
        cm_df
        .pivot(
            index="label",
            columns="prediction",
            values="count"
        )
        .fillna(0)
    )

    matrix = matrix.sort_index(
        axis=0
    ).sort_index(
        axis=1
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        matrix.values
    )

    plt.colorbar()

    plt.xticks(
        range(len(matrix.columns)),
        matrix.columns
    )

    plt.yticks(
        range(len(matrix.index)),
        matrix.index
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    plt.title(
        title
    )

    for i in range(
        matrix.shape[0]
    ):

        for j in range(
            matrix.shape[1]
        ):

            plt.text(
                j,
                i,
                int(matrix.iloc[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            filename
        ),
        dpi=300
    )

    plt.close()


# ================================================================
# 26. CREATE CONFUSION MATRIX FIGURES
# ================================================================

create_confusion_plot(
    lr_cm,
    "confusion_matrix_lr.png",
    "Logistic Regression - Confusion Matrix"
)

create_confusion_plot(
    rf_cm,
    "confusion_matrix_rf.png",
    "Random Forest - Confusion Matrix"
)


# ================================================================
# 27. BIG DATA SUMMARY
# ================================================================

summary = pd.DataFrame({
    "Metric": [
        "Dataset Size",
        "Total Records",
        "Total Attributes",
        "Spark Version",
        "Balanced ML Samples",
        "Training Samples",
        "Testing Samples"
    ],
    "Value": [
        "5 GB",
        f"{total_records:,}",
        len(df.columns),
        spark.version,
        f"{balanced_count:,}",
        f"{train_df.count():,}",
        f"{test_df.count():,}"
    ]
})

summary.to_csv(
    os.path.join(
        RESULTS_DIR,
        "big_data_summary.csv"
    ),
    index=False
)


# ================================================================
# 28. FINAL MESSAGE
# ================================================================

print("\n" + "=" * 70)
print("RESULT FILES CREATED")
print("=" * 70)

for filename in sorted(
    os.listdir(RESULTS_DIR)
):

    print(
        os.path.join(
            RESULTS_DIR,
            filename
        )
    )


# ================================================================
# 29. CLEANUP
# ================================================================

try:

    balanced_df.unpersist()
    train_df.unpersist()
    test_df.unpersist()
    processed_df.unpersist()

except Exception:
    pass


spark.stop()

print("\n" + "=" * 70)
print("RESULT GENERATION COMPLETED")
print("=" * 70)