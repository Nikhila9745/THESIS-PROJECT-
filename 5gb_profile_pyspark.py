# ============================================================
# BIG DATA ANALYTICS FOR PREDICTING STUDENT STRESS
# PySpark - 5 GB Dataset
# ============================================================

# ============================================================
# 2. IMPORT PYSPARK
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# ============================================================
# 3. START SPARK
# ============================================================

print("\n" + "=" * 60)
print("STARTING APACHE SPARK")
print("=" * 60)

spark = (
    SparkSession.builder
    .appName("StudentStress_5GB_Profile")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "16")
    .config("spark.sql.files.maxPartitionBytes", "128m")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("Spark started successfully!")
print("Spark version:", spark.version)

# ============================================================
# 4. DATASET PATH
# ============================================================

file_path = "/mnt/d/THESIS PROJECT/Stress_Data_5GB.csv"

print("\n" + "=" * 60)
print("DATASET")
print("=" * 60)

print("File:", file_path)

# ============================================================
# 5. LOAD 5 GB CSV USING PYSPARK
# ============================================================

print("\nLoading 5 GB dataset...")
print("Please wait...")

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("mode", "PERMISSIVE")
    .csv(file_path)
)

print("Dataset loaded successfully!")

# ============================================================
# 6. DATASET DIMENSIONS
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

row_count = df.count()
column_count = len(df.columns)

print("Number of rows    :", row_count)
print("Number of columns :", column_count)

# ============================================================
# 7. COLUMN NAMES
# ============================================================

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")

# ============================================================
# 8. SPARK DATAFRAME SCHEMA
# ============================================================

print("\n" + "=" * 60)
print("SPARK DATAFRAME SCHEMA")
print("=" * 60)

df.printSchema()

# ============================================================
# 9. FIRST 5 RECORDS
# ============================================================

print("\n" + "=" * 60)
print("FIRST 5 RECORDS")
print("=" * 60)

df.show(5, truncate=False)

# ============================================================
# 10. MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE CHECK")
print("=" * 60)

from pyspark.sql.functions import sum as spark_sum

missing_values = df.select(
    [
        spark_sum(
            col(c).isNull().cast("int")
        ).alias(c)
        for c in df.columns
    ]
)

missing_values.show(truncate=False)

# ============================================================
# 11. STRESS LEVEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("STRESS LEVEL DISTRIBUTION")
print("=" * 60)

stress_distribution = (
    df.groupBy("Stress_Level")
    .count()
    .orderBy(col("count").desc())
)

stress_distribution.show()

# ============================================================
# 12. STRESS LEVEL PERCENTAGE
# ============================================================

print("\n" + "=" * 60)
print("STRESS LEVEL PERCENTAGE")
print("=" * 60)

total = row_count

stress_percentage = (
    stress_distribution
    .withColumn(
        "Percentage",
        (col("count") / total * 100)
    )
)

stress_percentage.show()

# ============================================================
# 13. DATA TYPES SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

for field in df.schema.fields:
    print(f"{field.name:25} {field.dataType}")

# ============================================================
# 14. SPARK PARTITION INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("SPARK PARTITION INFORMATION")
print("=" * 60)

print(
    "Number of partitions:",
    df.rdd.getNumPartitions()
)

# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PYSPARK 5 GB DATA PROCESSING COMPLETED")
print("=" * 60)

print("Dataset              : Stress_Data_5GB.csv")
print("Rows                 :", row_count)
print("Columns              :", column_count)
print("Spark Version        :", spark.version)
print(
    "Spark Partitions     :",
    df.rdd.getNumPartitions()
)

print("\nMongoDB → Python baseline : COMPLETED")
print("PySpark 5 GB processing   : COMPLETED")

# ============================================================
# 16. STOP SPARK
# ============================================================

spark.stop()

print("\nSpark session stopped successfully.")
print("=" * 60)