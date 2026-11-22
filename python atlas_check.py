import os
from pyspark.sql import SparkSession

uri = os.environ["mongodb+srv://atlas_check"]

spark = (
    SparkSession.builder
    .appName("atlas_check")
    .master("local[2]")
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.13:10.5.0")
    .getOrCreate()
)

df = (
    spark.read.format("mongodb")
    .option("spark.mongodb.read.connection.uri", uri)
    .option("spark.mongodb.read.database", "student_stress_db")
    .option("spark.mongodb.read.collection", "stress_15gb")
    .load()
)

print("Columns:", len(df.columns))
print("Records in first 1000 rows:", df.limit(1000).count())
df.limit(3).show(truncate=False)

spark.stop()