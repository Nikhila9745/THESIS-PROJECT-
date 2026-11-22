# =============================================================================
# LINE-BY-LINE EXPLAINED VERSION of: student_stress_pyspark_ml.py
# The code is unchanged (extra blank lines inside blocks were removed for readability).
# Every line or block has a comment above it saying what it does.
#
# WHAT THIS SCRIPT DOES IN ONE PARAGRAPH
# It loads the 5 GB student-stress CSV with Spark, prints the class distribution and missing
# values, builds a preprocessing pipeline (fill missing numbers, encode categories, scale),
# removes duplicates, takes a class-balanced sample (max 10,000 per class), splits it 80/20,
# trains Logistic Regression and Random Forest, prints accuracy / precision / recall / F1 and a
# confusion matrix for each, and prints the most important feature positions of the forest.
# =============================================================================

# ================================================================
# BIG DATA ANALYTICS FOR STUDENT STRESS PREDICTION
# Memory-Optimized PySpark ML Pipeline
# Dataset: 5 GB Student Stress Dataset
# ================================================================

# os: environment variables and file checks. sys: exit the program. traceback: imported but never used.
import os
import sys
import traceback

# ================================================================
# 1. JAVA CONFIGURATION
# ================================================================

# Location of Java 17 on Windows (Spark runs on Java). The r"..." prefix stops backslashes being treated as escapes.
JAVA_HOME = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"

# Tell this process where Java is.
os.environ["JAVA_HOME"] = JAVA_HOME
# Put Java's bin folder at the FRONT of PATH so this Java is found first.
os.environ["PATH"] = os.path.join(JAVA_HOME, "bin") + os.pathsep + os.environ["PATH"]

# Print a banner line of 70 "=" characters.
print("=" * 70)
# Section title.
print("JAVA CONFIGURATION")
# Closing banner line.
print("=" * 70)
# Show the Java location being used.
print("JAVA_HOME:", os.environ["JAVA_HOME"])


# ================================================================
# 2. IMPORT PYSPARK
# ================================================================

# Imports are inside try so a missing PySpark gives a clear message instead of a long error.
try:
    # SparkSession = entry point to Spark.
    from pyspark.sql import SparkSession
    # functions (F) = column operations.
    from pyspark.sql import functions as F
    # DoubleType is imported but not used later.
    from pyspark.sql.types import DoubleType
    # Pipeline chains preprocessing stages.
    from pyspark.ml import Pipeline
    # Preprocessing tools.
    from pyspark.ml.feature import (
        # text -> number
        StringIndexer,
        # number -> 0/1 columns
        OneHotEncoder,
        # many columns -> one vector
        VectorAssembler,
        # rescale features
        StandardScaler,
        # fill missing numbers
        Imputer
    )
    # The two models.
    from pyspark.ml.classification import (
        LogisticRegression,
        RandomForestClassifier
    )
    # Computes accuracy / precision / recall / F1.
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    # StorageLevel lets us choose where cached data lives (here: disk).
    from pyspark.storagelevel import StorageLevel

# If any import fails:
except Exception as e:
    # Say what happened,
    print("\nERROR: PySpark import failed.")
    # show the error text,
    print(e)
    # and quit with a failure code.
    sys.exit(1)


# ================================================================
# 3. CREATE SPARK SESSION
# ================================================================

# "\n" prints a blank line first, then 70 "=".
print("\n" + "=" * 70)
print("STARTING APACHE SPARK")
print("=" * 70)

# Build the Spark session step by step.
spark = (
    SparkSession.builder
    # Name shown in Spark's UI.
    .appName("StudentStress_5GB_MemoryOptimized")
    # Run on this computer using all CPU cores.
    .master("local[*]")

    # Safer memory configuration
    # 10 GB for the driver (in local mode this is the memory that actually matters).
    .config("spark.driver.memory", "10g")
    # 10 GB for executors (has little effect in local mode, where the driver does the work).
    .config("spark.executor.memory", "10g")

    # Avoid excessive shuffle memory
    # Use 16 partitions after shuffles (default is 200, too many for one machine).
    .config("spark.sql.shuffle.partitions", "16")

    # CSV partition size
    # Split the CSV into ~128 MB chunks when reading.
    .config("spark.sql.files.maxPartitionBytes", "128m")

    # Reduce unnecessary console output
    # Show the progress bar in the console (note: "true" turns it ON, despite the comment above).
    .config("spark.ui.showConsoleProgress", "true")

    # Create the session (or reuse an existing one).
    .getOrCreate()
)

# Only show warnings and errors.
spark.sparkContext.setLogLevel("WARN")

# Print the Spark version in use.
print("Spark Version:", spark.version)


# ================================================================
# 4. DATASET PATH
# ================================================================

# Windows path of the 5 GB CSV (raw string).
DATASET_PATH = r"D:\THESIS PROJECT\Stress_Data_5GB.csv"

print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)
# Show the path being used.
print("Path:", DATASET_PATH)

# If the file does not exist:
if not os.path.exists(DATASET_PATH):
    print("\nERROR: Dataset not found!")
    print("Check this path:")
    print(DATASET_PATH)
    # Stop Spark cleanly,
    spark.stop()
    # and exit with a failure code.
    sys.exit(1)

# Reached only if the file exists.
print("Dataset found successfully.")


# ================================================================
# 5. LOAD FULL 5 GB DATASET
# ================================================================

print("\n" + "=" * 70)
print("LOADING FULL 5 GB DATASET USING PYSPARK")
print("=" * 70)

# try/except so a read error produces a clean message.
try:

    # Build the reader.
    df = (
        spark.read
        # First line of the CSV is the header.
        .option("header", "true")
        # Let Spark guess each column's type (makes an extra pass over the data).
        .option("inferSchema", "true")
        # Format to use if a column is parsed as a timestamp.
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
        # Read the file.
        .csv(DATASET_PATH)
    )

    # Show column names and the types Spark inferred.
    print("\nSchema:")
    df.printSchema()

    # Count all rows (this triggers a full read of the file).
    total_records = df.count()

    # Print row count with thousands separators, e.g. 12,345,678.
    print("\nTotal records:", f"{total_records:,}")
    # Print number of columns.
    print("Total columns:", len(df.columns))

    # Print how many partitions Spark split the data into (about 5 GB / 128 MB).
    print("\nSpark partitions:", df.rdd.getNumPartitions())


# On any loading error:
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

# Count rows per stress class and add each class's percentage.
distribution = (
    # Group by class,
    df.groupBy("Stress_Level")
    # count rows in each group (column named "count"),
    .count()
    # add a Percentage column = count * 100 / total, rounded to 2 decimals,
    .withColumn(
        "Percentage",
        F.round(F.col("count") * 100 / total_records, 2)
    )
    # biggest class first.
    .orderBy(F.desc("count"))
)

# Print the table without cutting off long text.
distribution.show(truncate=False)


# ================================================================
# 7. MISSING VALUE ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

# List of (column, number of missing values).
missing_values = []

# Check every column one at a time (each check re-scans the data, so this is slow on 5 GB).
for column in df.columns:

    # Count rows where this column is empty/null.
    null_count = df.filter(
        F.col(column).isNull()
    ).count()

    # Only remember columns that actually have missing values.
    if null_count > 0:

        missing_values.append(
            (column, null_count)
        )

# Nothing missing anywhere:
if len(missing_values) == 0:

    print("No missing values found.")

# Otherwise list the columns with missing values:
else:

    print("Columns containing missing values:")

    for column, count in missing_values:

        print(
            f"{column}: {count:,} missing values"
        )


# ================================================================
# 8. DEFINE FEATURES
# ================================================================

# Columns that hold numbers and will be used as features.
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


# Columns that hold categories (text) and will be one-hot encoded.
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

# Columns that must not be used as features:
#   Record_ID  = just an id; Stress_Score = the label is derived from it (leakage);
#   Comment    = free text; Timestamp = time of record.
drop_columns = [
    "Record_ID",
    "Stress_Score",
    "Comment",
    "Timestamp"
]

# Only drop the ones that really exist in this file (avoids an error if one is absent).
available_drop_columns = [
    c for c in drop_columns
    if c in df.columns
]

# Create the modelling table without those columns. "Stress_Level" is still present because it is needed for the label.
df_model = df.drop(*available_drop_columns)


# ================================================================
# 10. CLEAN NUMERIC FEATURES
# ================================================================

print("\n" + "=" * 70)
print("NUMERIC FEATURE IMPUTATION")
print("=" * 70)

# Keep only the numeric columns that exist in the table.
existing_numeric = [
    c for c in numeric_columns
    if c in df_model.columns
]

# Imputer fills missing numbers.
imputer = Imputer(
    # Columns to fix,
    inputCols=existing_numeric,
    # new columns with the filled values (suffix "_imputed"),
    outputCols=[
        c + "_imputed"
        for c in existing_numeric
    ],
    # fill with the median (robust to outliers).
    strategy="median"
)


# ================================================================
# 11. CLEAN CATEGORICAL FEATURES
# ================================================================

# For each category column that exists:
for column in categorical_columns:

    if column in df_model.columns:

        # Replace the column with a version where missing values become the text "Unknown".
        df_model = df_model.withColumn(
            column,
            # coalesce = first non-missing value of the list below.
            F.coalesce(
                # The value as text,
                F.col(column).cast("string"),
                # or "Unknown" if it is missing.
                F.lit("Unknown")
            )
        )


# ================================================================
# 12. CREATE TARGET LABEL
# ================================================================

print("\n" + "=" * 70)
print("CREATING TARGET LABEL")
print("=" * 70)

# StringIndexer turns "Low"/"Medium"/"High" into 0.0/1.0/2.0 in the column "label".
# NOTE: the numbers are assigned by FREQUENCY (most common class = 0.0), not by Low<Medium<High order.
label_indexer = StringIndexer(
    inputCol="Stress_Level",
    outputCol="label",
    # Unseen/odd values get an extra index instead of crashing.
    handleInvalid="keep"
)


# ================================================================
# 13. CATEGORICAL ENCODING
# ================================================================

# Category columns that exist in the table.
existing_categorical = [
    c for c in categorical_columns
    if c in df_model.columns
]

# One StringIndexer per category column: text -> number, in a new "<col>_index" column.
category_indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_index",
        handleInvalid="keep"
    )
    for c in existing_categorical
]

# One OneHotEncoder per category column: number -> 0/1 vector in "<col>_encoded".
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

# Names of the imputed numeric columns.
imputed_numeric = [
    c + "_imputed"
    for c in existing_numeric
]

# Names of the one-hot encoded category columns.
encoded_categorical = [
    c + "_encoded"
    for c in existing_categorical
]

# All columns that will form the feature vector: numbers first, then categories.
feature_columns = (
    imputed_numeric +
    encoded_categorical
)

# Combine them into ONE vector column called "features_raw".
assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features_raw",
    # "keep" does not drop rows with invalid values (NaN is kept in the vector).
    handleInvalid="keep"
)

# Rescale features so that each has unit standard deviation.
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    # Do not subtract the mean (keeps sparse one-hot vectors sparse),
    withMean=False,
    # but do divide by the standard deviation.
    withStd=True
)


# ================================================================
# 15. PREPROCESSING PIPELINE
# ================================================================

# Stages in order: fill numbers, then build the label.
preprocessing_stages = [
    imputer,
    label_indexer
]

# Add all category indexers.
preprocessing_stages.extend(category_indexers)
# Add all one-hot encoders.
preprocessing_stages.extend(category_encoders)
# Finally assemble the vector and scale it.
preprocessing_stages.extend([
    assembler,
    scaler
])

# Wrap the list of stages into one Pipeline object.
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

    # fit(): learn medians, category lists, label order and scaling factors from the FULL dataset.
    preprocessing_model = preprocessing_pipeline.fit(
        df_model
    )

    # transform(): apply them, producing the "label" and "features" columns.
    processed_df = preprocessing_model.transform(
        df_model
    )

    print("Preprocessing completed successfully.")

# On failure: show the error, stop Spark, exit.
except Exception as e:

    print("\nERROR during preprocessing:")
    print(e)

    spark.stop()
    sys.exit(1)


# ================================================================
# 17. KEEP ONLY REQUIRED COLUMNS
# ================================================================

# Keep just the label and the feature vector; everything else is no longer needed.
processed_df = processed_df.select(
    "label",
    "features"
)


# ================================================================
# 18. REMOVE DUPLICATES IF ANY
# ================================================================

# Remove rows whose (label, features) are identical to an earlier row.
processed_df = processed_df.dropDuplicates()


# ================================================================
# 19. PERSIST ONLY ON DISK
# ================================================================

# This avoids filling JVM memory with the complete 5 GB dataset.
# persist(DISK_ONLY): store the computed result on disk so later steps do not recompute everything.
processed_df = processed_df.persist(
    StorageLevel.DISK_ONLY
)

# count() forces the computation and stores the result.
processed_count = processed_df.count()

# Print the number of rows left after de-duplication.
print("\nProcessed records:", f"{processed_count:,}")


# ================================================================
# 20. LABEL DISTRIBUTION AFTER PREPROCESSING
# ================================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

# Rows per label, sorted by label value.
label_distribution = (
    processed_df
    .groupBy("label")
    .count()
    .orderBy("label")
)

# Print the table.
label_distribution.show()


# ================================================================
# 21. CREATE A CONTROLLED TRAINING SAMPLE
# ================================================================

print("\n" + "=" * 70)
print("CREATING CONTROLLED BALANCED TRAINING SAMPLE")
print("=" * 70)

# Determine counts for each class
# Turn the small table into a Python dictionary {label: row count}.
label_counts = {
    row["label"]: row["count"]
    for row in label_distribution.collect()
}

print("Available class counts:")
# Print the count of each label.
for label, count in label_counts.items():
    print(f"Label {label}: {count:,}")


# ------------------------------------------------
# Find minority class
# ------------------------------------------------

# With 2+ classes, the size of the smallest class limits how many examples per class we can balance to.
if len(label_counts) >= 2:

    minimum_class_count = min(
        label_counts.values()
    )

# With fewer than 2 classes, just use a default of 10,000.
else:

    minimum_class_count = 10000


# ------------------------------------------------
# Maximum examples per class
# ------------------------------------------------

# We deliberately limit the ML training size.
# This prevents Random Forest from exhausting
# the Java heap while preserving all 3 classes.

# At most 10,000 per class, or fewer if the smallest class is smaller.
TARGET_PER_CLASS = min(
    10000,
    minimum_class_count
)

# Print the chosen size.
print(
    "\nTarget samples per class:",
    TARGET_PER_CLASS
)


# ================================================================
# 22. BALANCED SAMPLING
# ================================================================

# Will hold one sampled DataFrame per class.
balanced_parts = []

# Go through each label in ascending order.
for label_value in sorted(label_counts.keys()):

    # All rows of this class.
    class_df = processed_df.filter(
        F.col("label") == float(label_value)
    )

    # How many rows this class has.
    class_count = label_counts[label_value]

    # If the class is bigger than the target, down-sample it:
    if class_count > TARGET_PER_CLASS:

        # Fraction of rows to keep.
        fraction = (
            TARGET_PER_CLASS /
            float(class_count)
        )

        sampled = (
            class_df
            # Random sample without replacement (the size is only approximate)...
            .sample(
                withReplacement=False,
                fraction=fraction,
                seed=42
            )
            # ...so cap it at exactly TARGET_PER_CLASS rows.
            .limit(TARGET_PER_CLASS)
        )

    # Otherwise keep the whole class.
    else:

        sampled = class_df

    # Store this class's sample.
    balanced_parts.append(sampled)


# Union all classes
# Start with the first class...
balanced_df = balanced_parts[0]

# ...and stack the other classes underneath it.
for part in balanced_parts[1:]:

    balanced_df = balanced_df.unionByName(part)


# Randomize sample order
# Shuffle the rows so the classes are mixed.
balanced_df = (
    balanced_df
    .orderBy(F.rand(seed=42))
)


# Store the balanced sample on disk.
balanced_df = balanced_df.persist(
    StorageLevel.DISK_ONLY
)

# Force computation and get its size.
balanced_count = balanced_df.count()

print(
    "\nBalanced training/evaluation dataset:",
    f"{balanced_count:,}"
)


# ================================================================
# 23. BALANCED DISTRIBUTION
# ================================================================

print("\nBalanced distribution:")

# Show the rows per label in the balanced sample (should be about equal).
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

# Random 80% / 20% split with a fixed seed.
train_df, test_df = balanced_df.randomSplit(
    [0.8, 0.2],
    seed=42
)

# Keep both parts on disk.
train_df = train_df.persist(
    StorageLevel.DISK_ONLY
)

test_df = test_df.persist(
    StorageLevel.DISK_ONLY
)

# Count rows in each part (also triggers the computation).
train_count = train_df.count()
test_count = test_df.count()

print("Training records:", f"{train_count:,}")
print("Testing records :", f"{test_count:,}")


# ================================================================
# 25. EVALUATION FUNCTION
# ================================================================

# Reusable function: given a model name and its predictions, print and return the metrics.
def evaluate_model(model_name, predictions):

    # Heading with the model name in capitals.
    print("\n")
    print("=" * 70)
    print(model_name.upper())
    print("=" * 70)

    # Four evaluators, one per metric.
    # Share of correct predictions.
    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )

    # Precision averaged over classes, weighted by class size.
    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )

    # Recall averaged over classes, weighted by class size.
    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedRecall"
    )

    # F1 averaged over classes, weighted by class size.
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    # Compute each metric from the predictions.
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

    # Print each metric to 4 decimal places.
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

    # Confusion matrix as a table: how many rows of each true label got each predicted label.
    print("\nConfusion Matrix:")

    (
        predictions
        .groupBy("label", "prediction")
        .count()
        .orderBy("label", "prediction")
        .show(20)
    )

    # How often each class was predicted (reveals a model that ignores a class).
    print("\nPrediction Distribution:")

    (
        predictions
        .groupBy("prediction")
        .count()
        .orderBy("prediction")
        .show()
    )

    # Return the four numbers so they can be printed again in the final summary.
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

# If training fails, the script continues to the Random Forest.
try:

    # Define the model.
    lr = LogisticRegression(
        # Input column (the scaled feature vector).
        featuresCol="features",
        # Column to predict.
        labelCol="label",

        # Reduced iterations for faster local execution
        maxIter=30,

        # Regularization
        # Strength of the penalty that stops the model overfitting.
        regParam=0.05,

        # 0.0 = pure L2 penalty (ridge-style).
        elasticNetParam=0.0,

        # One model that handles all 3 classes at once.
        family="multinomial"
    )

    # Train on the 80% training split.
    lr_model = lr.fit(train_df)

    print("Logistic Regression training completed.")

    # Predict on the 20% test split.
    lr_predictions = lr_model.transform(
        test_df
    )

    # Print and store the metrics.
    lr_results = evaluate_model(
        "Logistic Regression",
        lr_predictions
    )

# On error, print it and mark the result as missing.
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

    # If predictions exist, release them. (They were never persisted, so this is harmless but does nothing.)
    if "lr_predictions" in locals():

        lr_predictions.unpersist()

    # Delete the Logistic Regression model object.
    if "lr_model" in locals():

        del lr_model

    # Delete the Logistic Regression estimator object.
    if "lr" in locals():

        del lr

    # Force Spark garbage collection
    # Remove ALL cached data in Spark's cache (also the persisted train/test/balanced data, which will be recomputed when needed).
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

    # Define the forest.
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",

        # Much smaller than previous 100 trees
        # 30 trees.
        numTrees=30,

        # Much shallower than previous depth 12
        # Each tree may have at most 8 levels.
        maxDepth=8,

        # Limit feature search
        # Each split considers only sqrt(number of features) random features.
        featureSubsetStrategy="sqrt",

        # Reduce memory usage
        # Each tree is trained on a random 70% of the rows.
        subsamplingRate=0.7,

        # Fixed seed for repeatable results.
        seed=42
    )

    # Print the settings used (these are typed by hand, so keep them in sync with the values above).
    print("Random Forest configuration:")
    print("  Trees      :", 30)
    print("  Max Depth  :", 8)
    print("  Subsampling:", 0.7)
    print("  Features   : sqrt")

    # Train on the training split.
    rf_model = rf.fit(train_df)

    print(
        "\nRandom Forest training completed successfully."
    )

    # Predict on the test split.
    rf_predictions = rf_model.transform(
        test_df
    )

    # Print and store the metrics.
    rf_results = evaluate_model(
        "Random Forest",
        rf_predictions
    )

# If the forest fails (e.g. out of memory), explain and carry on.
except Exception as e:

    print("\nRandom Forest failed:")
    print(e)

    print("\nThe full Spark preprocessing still completed.")
    print("RF failure does not indicate a dataset or PySpark installation problem.")

    rf_results = None


# ================================================================
# 29. RANDOM FOREST FEATURE IMPORTANCE
# ================================================================

# Only if the forest was trained successfully.
if rf_results is not None:

    try:

        print("\n" + "=" * 70)
        print("RANDOM FOREST FEATURE IMPORTANCE")
        print("=" * 70)

        # One importance number per position in the feature vector (they add up to 1).
        importances = rf_model.featureImportances

        # How many positions there are (numeric columns + all one-hot columns).
        print(
            "Number of feature importance values:",
            len(importances)
        )

        # Print non-zero importance values
        non_zero = []

        # Loop over every position (i) and its importance (value).
        for i, value in enumerate(
            importances.toArray()
        ):

            # Keep only positions the forest actually used.
            if value > 0:

                non_zero.append(
                    (i, float(value))
                )

        # Sort from most to least important.
        non_zero.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # Positions are numbers, not names: 0..N-1 follow the order of feature_columns (numeric columns first, then one-hot columns).
        print("\nTop feature importance indices:")

        # Print the 20 most important positions.
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

# (Despite the section title, this only PRINTS the results; nothing is saved to a file.)
# Print the Logistic Regression summary if it succeeded.
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


# Print the Random Forest summary if it succeeded.
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

# Release the data stored on disk; ignore any error because we are finishing anyway.
try:

    balanced_df.unpersist()
    train_df.unpersist()
    test_df.unpersist()
    processed_df.unpersist()

except Exception:
    pass


# Shut down Spark and free all its resources.
spark.stop()

# Closing messages.
print("\n" + "=" * 70)
print("PYSPARK ML PIPELINE COMPLETED")
print("=" * 70)
print("Full 5 GB dataset was processed using Apache Spark.")
print("ML training used a controlled balanced sample.")
print("Random Forest configuration was memory optimized.")
print("=" * 70)
