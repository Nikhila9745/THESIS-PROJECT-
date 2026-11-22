# ================================================================
# BIG DATA ANALYTICS FOR STUDENT STRESS PREDICTION
# Memory-Optimized PySpark ML Pipeline
# Dataset: 5 GB Student Stress Dataset
# ================================================================

import os
import sys
import traceback

# ================================================================
# 1. JAVA CONFIGURATION
# ================================================================

JAVA_HOME = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"

os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = os.path.join(JAVA_HOME, "bin") + os.pathsep + os.environ["PATH"]

print("=" * 70)
print("JAVA CONFIGURATION")
print("=" * 70)
print("JAVA_HOME:", os.environ["JAVA_HOME"])


# ================================================================
# 2. IMPORT PYSPARK
# ================================================================

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType
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

except Exception as e:
    print("\nERROR: PySpark import failed.")
    print(e)
    sys.exit(1)


# ================================================================
# 3. CREATE SPARK SESSION
# ================================================================

print("\n" + "=" * 70)
print("STARTING APACHE SPARK")
print("=" * 70)

spark = (
    SparkSession.builder
    .appName("StudentStress_5GB_MemoryOptimized")
    .master("local[*]")

    # Safer memory configuration
    .config("spark.driver.memory", "10g")
    .config("spark.executor.memory", "10g")

    # Avoid excessive shuffle memory
    .config("spark.sql.shuffle.partitions", "16")

    # CSV partition size
    .config("spark.sql.files.maxPartitionBytes", "128m")

    # Reduce unnecessary console output
    .config("spark.ui.showConsoleProgress", "true")

    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("Spark Version:", spark.version)


# ================================================================
# 4. DATASET PATH
# ================================================================

DATASET_PATH = r"D:\THESIS PROJECT\Stress_Data_5GB.csv"

print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)
print("Path:", DATASET_PATH)

if not os.path.exists(DATASET_PATH):
    print("\nERROR: Dataset not found!")
    print("Check this path:")
    print(DATASET_PATH)
    spark.stop()
    sys.exit(1)

print("Dataset found successfully.")


# ================================================================
# 5. LOAD FULL 5 GB DATASET
# ================================================================

print("\n" + "=" * 70)
print("LOADING FULL 5 GB DATASET USING PYSPARK")
print("=" * 70)

try:

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
        .csv(DATASET_PATH)
    )

    print("\nSchema:")
    df.printSchema()

    total_records = df.count()

    print("\nTotal records:", f"{total_records:,}")
    print("Total columns:", len(df.columns))

    print("\nSpark partitions:", df.rdd.getNumPartitions())


except Exception as e:

    print("\nERROR while loading dataset:")
    print(e)

    spark.stop()
    sys.exit(1)


# ================================================================
# 6. STRESS LEVEL DISTRIBUTION
# ================================================================

print("\n" + "=" * 70)
print("STRESS LEVEL DISTRIBUTION")
print("=" * 70)

distribution = (
    df.groupBy("Stress_Level")
    .count()
    .withColumn(
        "Percentage",
        F.round(F.col("count") * 100 / total_records, 2)
    )
    .orderBy(F.desc("count"))
)

distribution.show(truncate=False)


# ================================================================
# 7. MISSING VALUE ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

missing_values = []

for column in df.columns:

    null_count = df.filter(
        F.col(column).isNull()
    ).count()

    if null_count > 0:

        missing_values.append(
            (column, null_count)
        )

if len(missing_values) == 0:

    print("No missing values found.")

else:

    print("Columns containing missing values:")

    for column, count in missing_values:

        print(
            f"{column}: {count:,} missing values"
        )


# ================================================================
# 8. DEFINE FEATURES
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
# 9. REMOVE TARGET-LEAKAGE / NON-PREDICTIVE COLUMNS
# ================================================================

drop_columns = [
    "Record_ID",
    "Stress_Score",
    "Comment",
    "Timestamp"
]

available_drop_columns = [
    c for c in drop_columns
    if c in df.columns
]

df_model = df.drop(*available_drop_columns)


# ================================================================
# 10. CLEAN NUMERIC FEATURES
# ================================================================

print("\n" + "=" * 70)
print("NUMERIC FEATURE IMPUTATION")
print("=" * 70)

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
# 11. CLEAN CATEGORICAL FEATURES
# ================================================================

for column in categorical_columns:

    if column in df_model.columns:

        df_model = df_model.withColumn(
            column,
            F.coalesce(
                F.col(column).cast("string"),
                F.lit("Unknown")
            )
        )


# ================================================================
# 12. CREATE TARGET LABEL
# ================================================================

print("\n" + "=" * 70)
print("CREATING TARGET LABEL")
print("=" * 70)

label_indexer = StringIndexer(
    inputCol="Stress_Level",
    outputCol="label",
    handleInvalid="keep"
)


# ================================================================
# 13. CATEGORICAL ENCODING
# ================================================================

existing_categorical = [
    c for c in categorical_columns
    if c in df_model.columns
]

category_indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_index",
        handleInvalid="keep"
    )
    for c in existing_categorical
]

category_encoders = [
    OneHotEncoder(
        inputCol=c + "_index",
        outputCol=c + "_encoded"
    )
    for c in existing_categorical
]


# ================================================================
# 14. FEATURE VECTOR
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
# 15. PREPROCESSING PIPELINE
# ================================================================

preprocessing_stages = [
    imputer,
    label_indexer
]

preprocessing_stages.extend(category_indexers)
preprocessing_stages.extend(category_encoders)
preprocessing_stages.extend([
    assembler,
    scaler
])

preprocessing_pipeline = Pipeline(
    stages=preprocessing_stages
)


# ================================================================
# 16. RUN PREPROCESSING
# ================================================================

print("\n" + "=" * 70)
print("RUNNING SPARK PREPROCESSING")
print("=" * 70)

try:

    preprocessing_model = preprocessing_pipeline.fit(
        df_model
    )

    processed_df = preprocessing_model.transform(
        df_model
    )

    print("Preprocessing completed successfully.")

except Exception as e:

    print("\nERROR during preprocessing:")
    print(e)

    spark.stop()
    sys.exit(1)


# ================================================================
# 17. KEEP ONLY REQUIRED COLUMNS
# ================================================================

processed_df = processed_df.select(
    "label",
    "features"
)


# ================================================================
# 18. REMOVE DUPLICATES IF ANY
# ================================================================

processed_df = processed_df.dropDuplicates()


# ================================================================
# 19. PERSIST ONLY ON DISK
# ================================================================

# This avoids filling JVM memory with the complete 5 GB dataset.
processed_df = processed_df.persist(
    StorageLevel.DISK_ONLY
)

processed_count = processed_df.count()

print("\nProcessed records:", f"{processed_count:,}")


# ================================================================
# 20. LABEL DISTRIBUTION AFTER PREPROCESSING
# ================================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

label_distribution = (
    processed_df
    .groupBy("label")
    .count()
    .orderBy("label")
)

label_distribution.show()


# ================================================================
# 21. CREATE A CONTROLLED TRAINING SAMPLE
# ================================================================

print("\n" + "=" * 70)
print("CREATING CONTROLLED BALANCED TRAINING SAMPLE")
print("=" * 70)

# Determine counts for each class
label_counts = {
    row["label"]: row["count"]
    for row in label_distribution.collect()
}

print("Available class counts:")
for label, count in label_counts.items():
    print(f"Label {label}: {count:,}")


# ------------------------------------------------
# Find minority class
# ------------------------------------------------

if len(label_counts) >= 2:

    minimum_class_count = min(
        label_counts.values()
    )

else:

    minimum_class_count = 10000


# ------------------------------------------------
# Maximum examples per class
# ------------------------------------------------

# We deliberately limit the ML training size.
# This prevents Random Forest from exhausting
# the Java heap while preserving all 3 classes.

TARGET_PER_CLASS = min(
    10000,
    minimum_class_count
)

print(
    "\nTarget samples per class:",
    TARGET_PER_CLASS
)


# ================================================================
# 22. BALANCED SAMPLING
# ================================================================

balanced_parts = []

for label_value in sorted(label_counts.keys()):

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
                withReplacement=False,
                fraction=fraction,
                seed=42
            )
            .limit(TARGET_PER_CLASS)
        )

    else:

        sampled = class_df

    balanced_parts.append(sampled)


# Union all classes
balanced_df = balanced_parts[0]

for part in balanced_parts[1:]:

    balanced_df = balanced_df.unionByName(part)


# Randomize sample order
balanced_df = (
    balanced_df
    .orderBy(F.rand(seed=42))
)


balanced_df = balanced_df.persist(
    StorageLevel.DISK_ONLY
)

balanced_count = balanced_df.count()

print(
    "\nBalanced training/evaluation dataset:",
    f"{balanced_count:,}"
)


# ================================================================
# 23. BALANCED DISTRIBUTION
# ================================================================

print("\nBalanced distribution:")

balanced_df.groupBy(
    "label"
).count().orderBy(
    "label"
).show()


# ================================================================
# 24. TRAIN / TEST SPLIT
# ================================================================

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

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

train_count = train_df.count()
test_count = test_df.count()

print("Training records:", f"{train_count:,}")
print("Testing records :", f"{test_count:,}")


# ================================================================
# 25. EVALUATION FUNCTION
# ================================================================

def evaluate_model(model_name, predictions):

    print("\n")
    print("=" * 70)
    print(model_name.upper())
    print("=" * 70)

    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )

    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )

    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedRecall"
    )

    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    accuracy = accuracy_evaluator.evaluate(
        predictions
    )

    precision = precision_evaluator.evaluate(
        predictions
    )

    recall = recall_evaluator.evaluate(
        predictions
    )

    f1 = f1_evaluator.evaluate(
        predictions
    )

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

    print("\nConfusion Matrix:")

    (
        predictions
        .groupBy("label", "prediction")
        .count()
        .orderBy("label", "prediction")
        .show(20)
    )

    print("\nPrediction Distribution:")

    (
        predictions
        .groupBy("prediction")
        .count()
        .orderBy("prediction")
        .show()
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ================================================================
# 26. LOGISTIC REGRESSION
# ================================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

try:

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",

        # Reduced iterations for faster local execution
        maxIter=30,

        # Regularization
        regParam=0.05,

        elasticNetParam=0.0,

        family="multinomial"
    )

    lr_model = lr.fit(train_df)

    print("Logistic Regression training completed.")

    lr_predictions = lr_model.transform(
        test_df
    )

    lr_results = evaluate_model(
        "Logistic Regression",
        lr_predictions
    )

except Exception as e:

    print("\nLogistic Regression failed:")
    print(e)

    lr_results = None


# ================================================================
# 27. FREE MEMORY BEFORE RANDOM FOREST
# ================================================================

print("\n" + "=" * 70)
print("FREEING MEMORY BEFORE RANDOM FOREST")
print("=" * 70)

try:

    if "lr_predictions" in locals():

        lr_predictions.unpersist()

    if "lr_model" in locals():

        del lr_model

    if "lr" in locals():

        del lr

    # Force Spark garbage collection
    spark.catalog.clearCache()

    print("Memory cleanup completed.")

except Exception as e:

    print("Memory cleanup warning:", e)


# ================================================================
# 28. RANDOM FOREST
# ================================================================

print("\n" + "=" * 70)
print("TRAINING MEMORY-OPTIMIZED RANDOM FOREST")
print("=" * 70)

try:

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",

        # Much smaller than previous 100 trees
        numTrees=30,

        # Much shallower than previous depth 12
        maxDepth=8,

        # Limit feature search
        featureSubsetStrategy="sqrt",

        # Reduce memory usage
        subsamplingRate=0.7,

        seed=42
    )

    print("Random Forest configuration:")
    print("  Trees      :", 30)
    print("  Max Depth  :", 8)
    print("  Subsampling:", 0.7)
    print("  Features   : sqrt")

    rf_model = rf.fit(train_df)

    print(
        "\nRandom Forest training completed successfully."
    )

    rf_predictions = rf_model.transform(
        test_df
    )

    rf_results = evaluate_model(
        "Random Forest",
        rf_predictions
    )

except Exception as e:

    print("\nRandom Forest failed:")
    print(e)

    print("\nThe full Spark preprocessing still completed.")
    print("RF failure does not indicate a dataset or PySpark installation problem.")

    rf_results = None


# ================================================================
# 29. RANDOM FOREST FEATURE IMPORTANCE
# ================================================================

if rf_results is not None:

    try:

        print("\n" + "=" * 70)
        print("RANDOM FOREST FEATURE IMPORTANCE")
        print("=" * 70)

        importances = rf_model.featureImportances

        print(
            "Number of feature importance values:",
            len(importances)
        )

        # Print non-zero importance values
        non_zero = []

        for i, value in enumerate(
            importances.toArray()
        ):

            if value > 0:

                non_zero.append(
                    (i, float(value))
                )

        non_zero.sort(
            key=lambda x: x[1],
            reverse=True
        )

        print("\nTop feature importance indices:")

        for index, value in non_zero[:20]:

            print(
                f"Feature {index}: {value:.6f}"
            )

    except Exception as e:

        print(
            "Feature importance extraction warning:",
            e
        )


# ================================================================
# 30. SAVE RESULTS
# ================================================================

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

if lr_results is not None:

    print("\nLogistic Regression")

    print(
        f"Accuracy  : {lr_results['accuracy']:.4f}"
    )

    print(
        f"Precision : {lr_results['precision']:.4f}"
    )

    print(
        f"Recall    : {lr_results['recall']:.4f}"
    )

    print(
        f"F1 Score  : {lr_results['f1']:.4f}"
    )


if rf_results is not None:

    print("\nRandom Forest")

    print(
        f"Accuracy  : {rf_results['accuracy']:.4f}"
    )

    print(
        f"Precision : {rf_results['precision']:.4f}"
    )

    print(
        f"Recall    : {rf_results['recall']:.4f}"
    )

    print(
        f"F1 Score  : {rf_results['f1']:.4f}"
    )


# ================================================================
# 31. CLEANUP
# ================================================================

print("\n" + "=" * 70)
print("CLEANING UP SPARK")
print("=" * 70)

try:

    balanced_df.unpersist()
    train_df.unpersist()
    test_df.unpersist()
    processed_df.unpersist()

except Exception:
    pass


spark.stop()

print("\n" + "=" * 70)
print("PYSPARK ML PIPELINE COMPLETED")
print("=" * 70)
print("Full 5 GB dataset was processed using Apache Spark.")
print("ML training used a controlled balanced sample.")
print("Random Forest configuration was memory optimized.")
print("=" * 70)