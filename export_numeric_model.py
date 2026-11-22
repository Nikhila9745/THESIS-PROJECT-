# export_numeric_model.py
# Trains a logistic regression on the numeric attributes only, then exports it as JSON
# so the website can compute predictions in the browser.
# Categorical attributes are left out, so this model is less accurate than the full pipeline.
#   python3 export_numeric_model.py --csv "/mnt/d/THESIS PROJECT/Stress_Data_5GB.csv" --out "/mnt/c/Users/JP/THESIS PROJECT/website/model.json"

import os
import json
import argparse
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import NumericType
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--per-class", type=int, default=10000)
args = parser.parse_args()

EXCLUDE = ["Record_ID", "Stress_Score", "Timestamp", "Stress_Level", "label"]
LABEL_NAMES = ["Low", "Medium", "High"]

spark = (SparkSession.builder.appName("Export numeric model")
         .master("local[*]").config("spark.driver.memory", "6g")
         .config("spark.sql.shuffle.partitions", "16").getOrCreate())
spark.sparkContext.setLogLevel("WARN")

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

counts = {r["label"]: r["count"] for r in df.groupBy("label").count().collect()}
fractions = {k: min(1.0, args.per_class / v) for k, v in counts.items()}
balanced = df.sampleBy("label", fractions=fractions, seed=42).dropDuplicates().cache()

# Ranges and medians for the form defaults
ranges = balanced.agg(*[F.min(c).alias(c + "_min") for c in numeric],
                      *[F.max(c).alias(c + "_max") for c in numeric],
                      *[F.expr(f"percentile_approx({c}, 0.5)").alias(c + "_median") for c in numeric]).first()

prep_model = Pipeline(stages=[
    Imputer(inputCols=numeric, outputCols=[c + "_imp" for c in numeric], strategy="median"),
    VectorAssembler(inputCols=[c + "_imp" for c in numeric], outputCol="features_raw", handleInvalid="skip"),
    StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True),
]).fit(balanced)
train = prep_model.transform(balanced).select("features", "label")
lr_model = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50,
                              regParam=0.05, family="multinomial").fit(train)

scaler_model = prep_model.stages[-1]
coef = lr_model.coefficientMatrix.toArray().tolist()
intercept = lr_model.interceptVector.toArray().tolist()
model_json = {
    "classes": LABEL_NAMES,
    "features": numeric,
    "means": scaler_model.mean.toArray().tolist(),
    "stds": scaler_model.std.toArray().tolist(),
    "coefficients": coef,
    "intercepts": intercept,
    "form_defaults": {c: float(ranges[c + "_median"]) for c in numeric},
    "form_ranges": {c: [float(ranges[c + "_min"]), float(ranges[c + "_max"])] for c in numeric},
    "training_records": balanced.count(),
    "note": "Logistic regression on numeric attributes only. Trained on a balanced sample.",
}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(model_json, f, indent=2)
# The website loads this file from disk, where browsers block fetch(), so write it as JavaScript too.
js_path = os.path.join(os.path.dirname(args.out), "model_data.js")
with open(js_path, "w", encoding="utf-8") as f:
    f.write("window.STRESS_MODEL = " + json.dumps(model_json) + ";\n")
print("Model exported to", args.out, "and", js_path)

# Check: recompute Spark's probabilities for one training row from the exported coefficients.
import math
row = train.limit(1).collect()[0]
z = row["features"].toArray()
scores = [intercept[k] + sum(coef[k][i] * z[i] for i in range(len(z))) for k in range(len(LABEL_NAMES))]
m = max(scores)
manual = [math.exp(s_ - m) for s_ in scores]
total = sum(manual)
manual = [v / total for v in manual]
spark_probs = lr_model.transform(train.limit(1)).select("probability").collect()[0][0].toArray().tolist()
print("Check, Spark probabilities :", [round(v, 6) for v in spark_probs])
print("Check, from exported JSON  :", [round(v, 6) for v in manual])
spark.stop()
