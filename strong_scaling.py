# strong_scaling.py
# Strong scaling: the same job on the same file, with a different number of CPU cores.
# Run it once per core count, then compare the rows in strong_scaling.csv.
#   python3 strong_scaling.py --csv "/mnt/d/THESIS PROJECT/Stress_Data_5GB.csv" --cores 1 --out "/mnt/c/Users/JP/THESIS PROJECT/scaling_5GB"
#   python3 strong_scaling.py --csv ... --cores 2 --out ...   (and 4, then the full machine)

import os
import csv
import time
import argparse
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import NumericType
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--cores", required=True, help="number of cores, or 'all'")
parser.add_argument("--out", required=True)
parser.add_argument("--driver-memory", default="6g")
args = parser.parse_args()

master = "local[*]" if args.cores == "all" else f"local[{args.cores}]"
spark = (SparkSession.builder.appName("Strong scaling")
         .master(master)
         .config("spark.driver.memory", args.driver_memory)
         .config("spark.sql.shuffle.partitions", str(max(4, int(args.cores) * 4) if args.cores != "all" else 16))
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")
cores_used = spark.sparkContext.defaultParallelism

EXCLUDE = ["Record_ID", "Stress_Score", "Timestamp", "Stress_Level", "label"]
t0 = time.time()
raw = spark.read.option("header", "true").option("inferSchema", "true").csv(args.csv)
df = raw.withColumn("label", F.when(F.col("Stress_Level") == "Low", 0.0)
                             .when(F.col("Stress_Level") == "Medium", 1.0)
                             .when(F.col("Stress_Level") == "High", 2.0))
df = df.filter(F.col("label").isNotNull())
numeric = [f.name for f in df.schema.fields
           if isinstance(f.dataType, NumericType) and f.name not in EXCLUDE]
for c in numeric:
    df = df.withColumn(c, F.col(c).cast("double"))
df = df.select(*numeric, "label")

# Same balanced sample size as the main pipeline (10,000 per class)
counts = {r["label"]: r["count"] for r in df.groupBy("label").count().collect()}
fractions = {k: min(1.0, 10000 / v) for k, v in counts.items()}
balanced = df.sampleBy("label", fractions=fractions, seed=42).cache()

prep = Pipeline(stages=[
    Imputer(inputCols=numeric, outputCols=[c + "_imp" for c in numeric]),
    VectorAssembler(inputCols=[c + "_imp" for c in numeric], outputCol="features_raw", handleInvalid="skip"),
    StandardScaler(inputCol="features_raw", outputCol="features", withMean=False, withStd=True),
]).fit(balanced)
train = prep.transform(balanced).select("features", "label").cache()
train.count()

t_fit = time.time()
RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=30, maxDepth=8,
                       subsamplingRate=0.7, featureSubsetStrategy="sqrt", seed=42).fit(train)
fit_seconds = time.time() - t_fit
total_seconds = time.time() - t0

os.makedirs(args.out, exist_ok=True)
path = os.path.join(args.out, "strong_scaling.csv")
new_file = not os.path.exists(path)
with open(path, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["requested_cores", "cores_used", "random_forest_fit_seconds",
                                      "total_seconds_including_load"])
    if new_file:
        w.writeheader()
    w.writerow({"requested_cores": args.cores, "cores_used": cores_used,
                "random_forest_fit_seconds": round(fit_seconds, 2),
                "total_seconds_including_load": round(total_seconds, 2)})
print(f"Cores used {cores_used}: RF fit {fit_seconds:.1f}s, total {total_seconds:.1f}s. Saved to {path}")
spark.stop()
