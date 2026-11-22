# =============================================================================
# LINE-BY-LINE EXPLAINED VERSION of: scalability_analysis.py
# The code is unchanged. Every line or block has a comment above it saying what it does.
#
# WHAT THIS SCRIPT DOES IN ONE PARAGRAPH
# It reads the run_summary.json that each experiment (5, 10, 15, 20 GB) saved, puts the key
# numbers (records, training time, F1) side by side in one table, saves the table as a CSV, and
# draws a chart of total runtime versus data size with the record count on a second axis.
# =============================================================================

# scalability_analysis.py
# Compares the 10GB, 15GB and 20GB runs using the run_summary.json and metrics files in each results folder.
# Only runs made on the same machine and settings are comparable.
# Run from the Ubuntu terminal:  python3 scalability_analysis.py

# os: file paths and folders.
import os
# json: read the run_summary.json files.
import json
# pandas: build the comparison table.
import pandas as pd
# matplotlib for the chart; "Agg" draws to a file without needing a screen.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# List of (label, results folder) pairs, one per experiment. (The header comment mentions 10/15/20 GB, but 5 GB is in the list too.)
RUNS = [
    ("5 GB", "/mnt/c/Users/JP/THESIS PROJECT/5GB_Results"),
    ("10 GB", "/mnt/c/Users/JP/THESIS PROJECT/10GB_Results"),
    ("15 GB", "/mnt/c/Users/JP/THESIS PROJECT/15GB_Results_with_Stress_Score"),
    ("20 GB", "/mnt/c/Users/JP/THESIS PROJECT/20GB_Results"),
]
# Folder where the table and chart are saved.
OUT_DIR = "/mnt/c/Users/JP/THESIS PROJECT/scalability_results"
# Create it if it is missing.
os.makedirs(OUT_DIR, exist_ok=True)

# Will hold one dictionary (one table row) per run found.
rows = []
# Go through each run.
for label, folder in RUNS:
    # Expected location of the summary file.
    summary_path = os.path.join(folder, "run_summary.json")
    # If it does not exist, say so and skip this run.
    if not os.path.exists(summary_path):
        print(f"Skipping {label}: no run_summary.json in {folder}")
        continue
    # Open the JSON file (UTF-8) ...
    with open(summary_path, encoding="utf-8") as f:
        # ... and load it into a Python dictionary.
        run = json.load(f)
    # "metrics" is a list of per-model results; turn it into {model name: its metrics} for easy lookup.
    metrics = {m["model"]: m for m in run.get("metrics", [])}
    # Logistic Regression metrics (empty dict if absent).
    lr = metrics.get("LogisticRegression", {})
    # Random Forest metrics (empty dict if absent).
    rf = metrics.get("RandomForest", {})
    # Add one row to the table; .get() returns None instead of crashing when a field is missing.
    rows.append({
        # Run label, e.g. "10 GB".
        "run": label,
        # Number of records left after cleaning.
        "records_after_cleaning": run.get("rows_after_cleaning"),
        # Size of the training set.
        "train_rows": run.get("train_rows"),
        # Size of the test set.
        "test_rows": run.get("test_rows"),
        # Total time for the whole pipeline.
        "total_seconds": run.get("total_seconds"),
        # Time to train Logistic Regression.
        "lr_train_seconds": lr.get("train_seconds"),
        # Time to train Random Forest.
        "rf_train_seconds": rf.get("train_seconds"),
        # Logistic Regression weighted F1.
        "lr_weighted_f1": lr.get("weighted_f1"),
        # Random Forest weighted F1.
        "rf_weighted_f1": rf.get("weighted_f1"),
        # How many test rows also appear in the training set (a duplicate/leakage check).
        "test_rows_also_in_train": run.get("test_rows_also_in_train"),
    })

# If no run had a summary file, stop with a clear message.
if not rows:
    raise SystemExit("No runs found. Check RUNS paths.")

# Convert the rows into a pandas table.
table = pd.DataFrame(rows)
# Only compute the normalised time if every run has a total_seconds value.
if "total_seconds" in table and table["total_seconds"].notna().all():
    # Seconds per million records = total time / (records in millions), rounded to 2 decimals. Roughly flat means linear scaling.
    table["seconds_per_million_records"] = (table["total_seconds"] /
                                            (table["records_after_cleaning"] / 1e6)).round(2)
# Save the table to CSV.
table.to_csv(os.path.join(OUT_DIR, "scalability_summary.csv"), index=False)
# Print the table in the terminal.
print(table.to_string(index=False))

# Plot: total runtime and records against data size.
# Create the figure with the first (left) y-axis.
fig, ax1 = plt.subplots(figsize=(7, 4.5))
# x positions 0, 1, 2, ... one per run.
x = range(len(table))
# Blue bars = total pipeline time.
ax1.bar(x, table["total_seconds"], color="#3B6EA8", alpha=0.8)
# Left axis label.
ax1.set_ylabel("Total pipeline time (seconds)")
# Put tick marks at each bar ...
ax1.set_xticks(list(x))
# ... and label them with the run names (5 GB, 10 GB, ...).
ax1.set_xticklabels(table["run"])
# Write the seconds value above each bar.
for i, v in enumerate(table["total_seconds"]):
    ax1.text(i, v, f"{v:.0f}s", ha="center", va="bottom", fontsize=8)
# Second y-axis sharing the same x-axis, for the record counts.
ax2 = ax1.twinx()
# Orange line with circles = records after cleaning, in millions.
ax2.plot(list(x), table["records_after_cleaning"] / 1e6, "o-", color="#D95F02")
# Right axis label in the same orange.
ax2.set_ylabel("Records after cleaning (millions)", color="#D95F02")
# Chart title.
ax1.set_title("Scalability: runtime and records by data size")
# Fit the layout.
fig.tight_layout()
# Save the chart as a 200-dpi PNG.
fig.savefig(os.path.join(OUT_DIR, "scalability_runtime.png"), dpi=200)
# Confirm where everything was saved.
print("Saved to:", OUT_DIR)
