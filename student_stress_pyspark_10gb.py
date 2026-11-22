# ============================================================
# BIG DATA ANALYTICS FOR STUDENT STRESS
# PySpark Processing - 10 GB Dataset
# ============================================================

import os
import time
import csv


# ------------------------------------------------------------
# 1. Configure Java
# ------------------------------------------------------------

JAVA_HOME = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"

os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = (
    os.path.join(JAVA_HOME, "bin")
    + os.pathsep
    + os.environ["PATH"]
)


# ------------------------------------------------------------
# 2. Import PySpark
# ------------------------------------------------------------

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    round as spark_round,
    sum as spark_sum
)


# ------------------------------------------------------------
# 3. Start Spark
# ------------------------------------------------------------

spark = (
    SparkSession.builder
    .appName("StudentStress_10GB")
    .master("local[*]")
    .config("spark.driver.memory", "10g")
    .config("spark.executor.memory", "10g")
    .config("spark.driver.maxResultSize", "2g")
    .config("spark.sql.shuffle.partitions", "16")
    .config("spark.sql.files.maxPartitionBytes", "128m")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ------------------------------------------------------------
# 4. Start timer
# ------------------------------------------------------------

start_time = time.time()


print("=" * 70)
print("        STUDENT STRESS - 10 GB PYSPARK PROCESSING")
print("=" * 70)


# ------------------------------------------------------------
# 5. Dataset path
# ------------------------------------------------------------

dataset_path = r"D:\THESIS PROJECT\Stress_Data_10GB.csv"

print("\nDataset:")
print(dataset_path)


# ------------------------------------------------------------
# 6. Load 10 GB CSV
# ------------------------------------------------------------

print("\nLoading 10 GB dataset...")

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
    .csv(dataset_path)
)

print("Dataset loaded successfully.")


# ------------------------------------------------------------
# 7. Dataset information
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

num_rows = df.count()
num_columns = len(df.columns)
num_partitions = df.rdd.getNumPartitions()

print(f"Total Records       : {num_rows:,}")
print(f"Total Attributes    : {num_columns}")
print(f"Spark Partitions    : {num_partitions}")


# ------------------------------------------------------------
# 8. Display schema
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA SCHEMA")
print("=" * 70)

df.printSchema()


# ------------------------------------------------------------
# 9. Stress-level distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STRESS-LEVEL DISTRIBUTION")
print("=" * 70)

stress_counts = (
    df.groupBy("Stress_Level")
    .count()
    .orderBy(col("count").desc())
)

stress_counts.show(truncate=False)


# ------------------------------------------------------------
# 10. Stress distribution with percentages
# ------------------------------------------------------------

stress_distribution = (
    stress_counts
    .withColumn(
        "Percentage",
        spark_round(
            (col("count") / num_rows) * 100,
            4
        )
    )
)

print("\nStress distribution with percentages:")

stress_distribution.show(truncate=False)


# ------------------------------------------------------------
# 11. Missing-value analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

missing_df = df.select([
    spark_sum(
        col(column_name).isNull().cast("int")
    ).alias(column_name)
    for column_name in df.columns
])

missing_values = missing_df.collect()[0].asDict()

print("\nColumns containing missing values:")

missing_columns = []

for column_name, missing_count in missing_values.items():

    if missing_count > 0:

        missing_columns.append(
            (column_name, missing_count)
        )

        print(
            f"{column_name:30s}: "
            f"{missing_count:,}"
        )

if len(missing_columns) == 0:
    print("No missing values found.")


# ------------------------------------------------------------
# 12. Identify numeric and categorical attributes
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ATTRIBUTE TYPES")
print("=" * 70)

numeric_columns = []
categorical_columns = []

for field in df.schema.fields:

    data_type = field.dataType.simpleString()

    if data_type in [
        "int",
        "bigint",
        "double",
        "float",
        "long",
        "short",
        "decimal"
    ]:
        numeric_columns.append(field.name)

    elif data_type == "string":
        categorical_columns.append(field.name)


# ------------------------------------------------------------
# 13. Numeric attributes
# ------------------------------------------------------------

print("\nNumeric Attributes:")

for column_name in numeric_columns:
    print(" -", column_name)

print(
    f"\nTotal Numeric Attributes: "
    f"{len(numeric_columns)}"
)


# ------------------------------------------------------------
# 14. Categorical attributes
# ------------------------------------------------------------

print("\nCategorical Attributes:")

for column_name in categorical_columns:
    print(" -", column_name)

print(
    f"\nTotal Categorical Attributes: "
    f"{len(categorical_columns)}"
)


# ------------------------------------------------------------
# 15. Target variable
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET VARIABLE")
print("=" * 70)

print("Target column: Stress_Level")

print("\nTarget classes:")

df.select("Stress_Level") \
    .distinct() \
    .orderBy("Stress_Level") \
    .show(truncate=False)


# ------------------------------------------------------------
# 16. Numeric feature summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NUMERIC FEATURE SUMMARY")
print("=" * 70)

if len(numeric_columns) > 0:

    df.select(numeric_columns).summary(
        "count",
        "mean",
        "stddev",
        "min",
        "max"
    ).show(truncate=False)


# ------------------------------------------------------------
# 17. Processing time
# ------------------------------------------------------------

end_time = time.time()

processing_time = end_time - start_time
processing_minutes = processing_time / 60


print("\n" + "=" * 70)
print("PROCESSING SUMMARY")
print("=" * 70)

print("Dataset Size       : 10 GB")
print(f"Total Records      : {num_rows:,}")
print(f"Total Attributes   : {num_columns}")
print(f"Spark Partitions   : {num_partitions}")
print(f"Processing Time    : {processing_time:.2f} seconds")
print(f"Processing Time    : {processing_minutes:.2f} minutes")


# ------------------------------------------------------------
# 18. Save summary using Python
# ------------------------------------------------------------

summary_file = (
    r"D:\THESIS PROJECT\pyspark_10GB_summary.csv"
)

summary_data = [
    ["Metric", "Value"],

    ["Dataset_Size_GB", "10"],

    ["Total_Records",
     str(num_rows)],

    ["Total_Attributes",
     str(num_columns)],

    ["Spark_Partitions",
     str(num_partitions)],

    ["Processing_Time_Seconds",
     f"{processing_time:.2f}"],

    ["Processing_Time_Minutes",
     f"{processing_minutes:.2f}"]
]


with open(
    summary_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerows(summary_data)


print("\nSummary saved to:")
print(summary_file)


# ------------------------------------------------------------
# 19. Finish
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("10 GB PYSPARK PROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)


# ------------------------------------------------------------
# 20. Stop Spark
# ------------------------------------------------------------

spark.stop()