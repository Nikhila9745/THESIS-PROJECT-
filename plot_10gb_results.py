# plot_10gb_results.py
# Draws the figures for the 10GB run from the results folder written by the 10GB Spark ML pipeline.
# Run from the Ubuntu terminal:  python3 plot_10gb_results.py
# To use a different results folder without editing this file:
#   RESULTS_DIR="/mnt/c/Users/JP/THESIS PROJECT/10GB_Results" python3 plot_10gb_results.py

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.environ.get("RESULTS_DIR", "/mnt/c/Users/JP/THESIS PROJECT/10GB_Results")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
CLASS_ORDER = ["Low", "Medium", "High"]
MODEL_NAMES = {"LogisticRegression": "Logistic Regression", "RandomForest": "Random Forest"}
TITLE_SIZE = 13

os.makedirs(PLOT_DIR, exist_ok=True)


def load(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"Missing file, skipping: {name}")
        return None
    return pd.read_csv(path)


def save(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("Saved", path)


def axis_limits(values, pad=0.05):
    # Set the y-axis from the data so low scores are not cut off, and never above 1.05.
    lo = max(0.0, min(values) - pad)
    return lo, 1.05


# 1. Model comparison on the balanced test set
metrics = load("metrics_summary.csv")
if metrics is not None:
    cols = [("accuracy", "Accuracy"), ("weighted_f1", "Weighted F1"),
            ("weighted_precision", "Weighted precision"), ("weighted_recall", "Weighted recall")]
    x = np.arange(len(cols))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (_, row) in enumerate(metrics.iterrows()):
        values = [row[c] for c, _ in cols]
        bars = ax.bar(x + (i - 0.5) * width, values, width,
                      label=MODEL_NAMES.get(row["model"], row["model"]))
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in cols])
    ax.set_ylim(*axis_limits(metrics[[c for c, _ in cols]].values.ravel()))
    ax.set_ylabel("Score")
    ax.set_title("10GB run: model comparison on the balanced test set", fontsize=TITLE_SIZE)
    ax.legend(loc="upper center", ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "10gb_model_comparison.png")

# 2. Per-class F1 for each model
per_class = load("per_class_metrics.csv")
if per_class is not None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(CLASS_ORDER))
    width = 0.38
    models = list(per_class["model"].unique())
    for i, model in enumerate(models):
        sub = per_class[per_class["model"] == model].set_index("class").reindex(CLASS_ORDER)
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, sub["f1"], width,
                      label=MODEL_NAMES.get(model, model))
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylim(*axis_limits(per_class["f1"].values))
    ax.set_ylabel("F1 score")
    ax.set_title("10GB run: per-class F1 on the balanced test set", fontsize=TITLE_SIZE)
    ax.legend(loc="upper center", ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "10gb_per_class_f1.png")

# 3. Confusion matrices (rows = true class, columns = predicted class)
for model in ["LogisticRegression", "RandomForest"]:
    cm = load(f"confusion_{model}.csv")
    if cm is None:
        continue
    matrix = (cm.pivot_table(index="label", columns="prediction", values="count", fill_value=0)
                .reindex(index=[0.0, 1.0, 2.0], columns=[0.0, 1.0, 2.0], fill_value=0))
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(matrix.values, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    for r in range(3):
        for c in range(3):
            value = int(matrix.values[r, c])
            color = "white" if matrix.values[r, c] > matrix.values.max() / 2 else "black"
            ax.text(c, r, f"{value:,}", ha="center", va="center", color=color)
    ax.set_title(f"10GB run: confusion matrix, {MODEL_NAMES.get(model, model)}", fontsize=TITLE_SIZE)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, f"10gb_confusion_{model}.png")

# 4. Top 15 Random Forest feature importances
fi = load("feature_importance_random_forest.csv")
if fi is not None:
    top = fi.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(top["feature"], top["importance"], color="#3B6EA8")
    ax.set_xlabel("Importance (Random Forest)")
    ax.set_title("10GB run: top 15 features", fontsize=TITLE_SIZE)
    ax.grid(axis="x", alpha=0.3)
    save(fig, "10gb_feature_importance_top15.png")

print("Done. Figures are in:", PLOT_DIR)