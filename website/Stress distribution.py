"""
Stress-level distribution for the 10 GB, 15 GB and 20 GB runs.

Counts Stress_Level per dataset with Spark, saves the counts and shares to CSV,
and draws a count chart and a percentage chart.

Usage (defaults point at the thesis folder; change the paths if yours differ):
    python stress_distribution.py
    python stress_distribution.py --out "D:\\THESIS PROJECT\\Distribution"
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SparkSession, functions as F

CLASS_ORDER = ["Low", "Medium", "High"]
DATASETS = {
    "10GB": r"D:\THESIS PROJECT\Stress_Data_10GB.csv",
    "15GB": r"D:\THESIS PROJECT\Stress_Data_15GB.csv",
    "20GB": r"D:\THESIS PROJECT\Stress_Data_20GB.csv",
}

parser = argparse.ArgumentParser()
parser.add_argument("--out", default=os.path.join(os.getcwd(), "stress_distribution_results"))
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

spark = (SparkSession.builder
         .appName("StressLevelDistribution")
         .config("spark.driver.memory", "8g")
         .getOrCreate())

rows = []
for label, path in DATASETS.items():
    if not os.path.isfile(path):
        print(f"[skip] {label}: file not found at {path}")
        continue
    df = (spark.read
          .option("header", "true")
          .option("inferSchema", "false")   # labels are read as text; no need to infer types
          .csv(path)
          .select("Stress_Level"))
    counts = {r["Stress_Level"]: r["count"]
              for r in df.groupBy("Stress_Level").count().collect()}
    total = sum(counts.values())
    for cls in CLASS_ORDER:
        n = int(counts.get(cls, 0))
        rows.append({"Dataset": label, "Stress_Level": cls, "Count": n,
                     "Percent": round(100.0 * n / total, 2) if total else 0.0,
                     "Total_records": total})
    print(f"{label}: total {total:,} records")

spark.stop()

result = pd.DataFrame(rows)
result.to_csv(os.path.join(args.out, "stress_distribution_counts.csv"), index=False)
print(result.to_string(index=False))

# Plot 1: counts (log scale, because the low class is very small)
datasets = list(result["Dataset"].unique())
fig, ax = plt.subplots(figsize=(9, 5))
width = 0.25
for i, cls in enumerate(CLASS_ORDER):
    sub = result[result["Stress_Level"] == cls].set_index("Dataset").reindex(datasets)
    xs = [j + (i - 1) * width for j in range(len(datasets))]
    bars = ax.bar(xs, sub["Count"], width, label=cls)
    for b, v in zip(bars, sub["Count"]):
        ax.annotate(f"{int(v):,}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
ax.set_yscale("log")
ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets)
ax.set_ylabel("Records (log scale)")
ax.set_title("Stress-level counts by dataset size")
ax.legend(title="Stress level")
fig.tight_layout()
fig.savefig(os.path.join(args.out, "fig_stress_counts.png"), dpi=200)
plt.close(fig)

# Plot 2: percentage share per dataset (stacked)
pivot = (result.pivot(index="Dataset", columns="Stress_Level", values="Percent")
         .reindex(index=datasets, columns=CLASS_ORDER))
fig, ax = plt.subplots(figsize=(8, 5))
bottom = [0.0] * len(datasets)
for cls in CLASS_ORDER:
    vals = pivot[cls].tolist()
    ax.bar(datasets, vals, bottom=bottom, label=cls)
    for x, b, v in zip(datasets, bottom, vals):
        if v > 2:
            ax.text(x, b + v / 2, f"{v:.2f}%", ha="center", va="center", fontsize=9, color="white")
    bottom = [b + v for b, v in zip(bottom, vals)]
ax.set_ylabel("Share of records (%)")
ax.set_title("Stress-level share by dataset size")
ax.legend(title="Stress level", loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(args.out, "fig_stress_shares.png"), dpi=200)
plt.close(fig)

print("Saved to", args.out)