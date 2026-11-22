# scalability_analysis.py
# Compares the 10GB, 15GB and 20GB runs using the run_summary.json and metrics files in each results folder.
# Only runs made on the same machine and settings are comparable.
# Run from the Ubuntu terminal:  python3 scalability_analysis.py

import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [
    ("5 GB", "/mnt/c/Users/JP/THESIS PROJECT/5GB_Results"),
    ("10 GB", "/mnt/c/Users/JP/THESIS PROJECT/10GB_Results"),
    ("15 GB", "/mnt/c/Users/JP/THESIS PROJECT/15GB_Results_with_Stress_Score"),
    ("20 GB", "/mnt/c/Users/JP/THESIS PROJECT/20GB_Results"),
]
OUT_DIR = "/mnt/c/Users/JP/THESIS PROJECT/scalability_results"
os.makedirs(OUT_DIR, exist_ok=True)

rows = []
for label, folder in RUNS:
    summary_path = os.path.join(folder, "run_summary.json")
    if not os.path.exists(summary_path):
        print(f"Skipping {label}: no run_summary.json in {folder}")
        continue
    with open(summary_path, encoding="utf-8") as f:
        run = json.load(f)
    metrics = {m["model"]: m for m in run.get("metrics", [])}
    lr = metrics.get("LogisticRegression", {})
    rf = metrics.get("RandomForest", {})
    rows.append({
        "run": label,
        "records_after_cleaning": run.get("rows_after_cleaning"),
        "train_rows": run.get("train_rows"),
        "test_rows": run.get("test_rows"),
        "total_seconds": run.get("total_seconds"),
        "lr_train_seconds": lr.get("train_seconds"),
        "rf_train_seconds": rf.get("train_seconds"),
        "lr_weighted_f1": lr.get("weighted_f1"),
        "rf_weighted_f1": rf.get("weighted_f1"),
        "test_rows_also_in_train": run.get("test_rows_also_in_train"),
    })

if not rows:
    raise SystemExit("No runs found. Check RUNS paths.")

table = pd.DataFrame(rows)
if "total_seconds" in table and table["total_seconds"].notna().all():
    table["seconds_per_million_records"] = (table["total_seconds"] /
                                            (table["records_after_cleaning"] / 1e6)).round(2)
table.to_csv(os.path.join(OUT_DIR, "scalability_summary.csv"), index=False)
print(table.to_string(index=False))

# Plot: total runtime and records against data size.
fig, ax1 = plt.subplots(figsize=(7, 4.5))
x = range(len(table))
ax1.bar(x, table["total_seconds"], color="#3B6EA8", alpha=0.8)
ax1.set_ylabel("Total pipeline time (seconds)")
ax1.set_xticks(list(x))
ax1.set_xticklabels(table["run"])
for i, v in enumerate(table["total_seconds"]):
    ax1.text(i, v, f"{v:.0f}s", ha="center", va="bottom", fontsize=8)
ax2 = ax1.twinx()
ax2.plot(list(x), table["records_after_cleaning"] / 1e6, "o-", color="#D95F02")
ax2.set_ylabel("Records after cleaning (millions)", color="#D95F02")
ax1.set_title("Scalability: runtime and records by data size")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "scalability_runtime.png"), dpi=200)
print("Saved to:", OUT_DIR)