# ============================================================
# BIG DATA ANALYTICS FOR PREDICTING STUDENT STRESS LEVELS
# MongoDB Atlas + Machine Learning
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from getpass import getpass
from pymongo import MongoClient
from urllib.parse import quote_plus

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. MONGODB ATLAS CONNECTION
# ============================================================

print("\n==========================================")
print("CONNECTING TO MONGODB ATLAS")
print("==========================================")

username = "admin"

password = getpass("Enter MongoDB Atlas password: ")

password_encoded = quote_plus(password)

uri = (
    f"mongodb://{username}:{password_encoded}@"
    "ac-nxcyuch-shard-00-00.bkbp3sp.mongodb.net:27017,"
    "ac-nxcyuch-shard-00-01.bkbp3sp.mongodb.net:27017,"
    "ac-nxcyuch-shard-00-02.bkbp3sp.mongodb.net:27017/"
    "?ssl=true"
    "&replicaSet=atlas-thfw0a-shard-0"
    "&authSource=admin"
    "&appName=Cluster0"
)

try:

    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=10000
    )

    client.admin.command("ping")

    print("MongoDB Atlas connection successful!")

except Exception as e:

    print("MongoDB connection failed.")
    print("Error:", e)

    raise SystemExit


# ============================================================
# 2. DATABASE AND COLLECTION
# ============================================================

db = client["student_stress_db"]

collection = db["stress_5gb"]

print("\nDatabase   :", db.name)
print("Collection :", collection.name)


# ============================================================
# 3. COUNT DOCUMENTS
# ============================================================

total_documents = collection.count_documents({})

print("\nTotal documents available:", total_documents)


# ============================================================
# 4. LOAD DATA FROM MONGODB
# ============================================================

sample_size = min(100000, total_documents)

print(
    "\nLoading",
    sample_size,
    "documents from MongoDB..."
)

cursor = collection.find(
    {},
    {"_id": 0}
).limit(sample_size)

df = pd.DataFrame(list(cursor))

print("Data loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# 5. BASIC DATASET INFORMATION
# ============================================================

print("\n==========================================")
print("DATASET INFORMATION")
print("==========================================")

print("\nFirst 5 records:")
print(df.head())

print("\nDataset dimensions:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 6. STRESS LEVEL DISTRIBUTION
# ============================================================

print("\n==========================================")
print("STRESS LEVEL DISTRIBUTION")
print("==========================================")

print(
    df["Stress_Level"].value_counts(dropna=False)
)

print("\nPercentage distribution:")

print(
    df["Stress_Level"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(2)
)


# ============================================================
# 7. REMOVE INVALID TARGET VALUES
# ============================================================

print("\nCleaning target variable...")

# Remove records where Stress_Level is missing.

df = df.dropna(
    subset=["Stress_Level"]
).copy()

# Convert target to string so that all target
# labels have a consistent datatype.

df["Stress_Level"] = (
    df["Stress_Level"]
    .astype(str)
    .str.strip()
)

print(
    "Records after target cleaning:",
    len(df)
)


# ============================================================
# 8. PREPARE FEATURES AND TARGET
# ============================================================

target = "Stress_Level"

drop_columns = [
    "Record_ID",
    "Stress_Score",
    "Comment",
    "Timestamp"
]

X = df.drop(
    columns=drop_columns + [target],
    errors="ignore"
).copy()

y = df[target].copy()


# ============================================================
# 9. IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=[
        "int64",
        "float64",
        "int32",
        "float32"
    ]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=[
        "object",
        "category",
        "bool"
    ]
).columns.tolist()


print("\n==========================================")
print("FEATURE TYPES")
print("==========================================")

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 10. CLEAN CATEGORICAL DATA TYPES
# ============================================================

# Convert categorical columns to strings.
#
# This prevents errors when a column contains a mixture
# of strings and numeric/NaN values.

for column in categorical_features:

    X[column] = (
        X[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )


# ============================================================
# 11. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n==========================================")
print("TRAIN / TEST SPLIT")
print("==========================================")

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples :",
    len(X_test)
)


# ============================================================
# 12. PREPROCESSING PIPELINE
# ============================================================

numeric_transformer = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[

        (
            "num",
            numeric_transformer,
            numeric_features
        ),

        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


print("\nPreprocessing pipeline created.")


# ============================================================
# 13. LOGISTIC REGRESSION
# ============================================================

print("\n==========================================")
print("LOGISTIC REGRESSION")
print("==========================================")

logistic_model = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

print("Training Logistic Regression...")

logistic_model.fit(
    X_train,
    y_train
)

print(
    "Logistic Regression training completed."
)

y_pred_lr = logistic_model.predict(
    X_test
)


# ============================================================
# 14. LOGISTIC REGRESSION RESULTS
# ============================================================

lr_accuracy = accuracy_score(
    y_test,
    y_pred_lr
)

lr_precision = precision_score(
    y_test,
    y_pred_lr,
    average="weighted",
    zero_division=0
)

lr_recall = recall_score(
    y_test,
    y_pred_lr,
    average="weighted",
    zero_division=0
)

lr_f1 = f1_score(
    y_test,
    y_pred_lr,
    average="weighted",
    zero_division=0
)

print("\nLogistic Regression Results")
print("--------------------------------")

print(
    f"Accuracy  : {lr_accuracy:.4f}"
)

print(
    f"Precision : {lr_precision:.4f}"
)

print(
    f"Recall    : {lr_recall:.4f}"
)

print(
    f"F1 Score  : {lr_f1:.4f}"
)


# ============================================================
# 15. RANDOM FOREST
# ============================================================

print("\n==========================================")
print("RANDOM FOREST")
print("==========================================")

random_forest_model = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("Training Random Forest...")

random_forest_model.fit(
    X_train,
    y_train
)

print(
    "Random Forest training completed."
)

y_pred_rf = random_forest_model.predict(
    X_test
)


# ============================================================
# 16. RANDOM FOREST RESULTS
# ============================================================

rf_accuracy = accuracy_score(
    y_test,
    y_pred_rf
)

rf_precision = precision_score(
    y_test,
    y_pred_rf,
    average="weighted",
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    y_pred_rf,
    average="weighted",
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    y_pred_rf,
    average="weighted",
    zero_division=0
)

print("\nRandom Forest Results")
print("--------------------------------")

print(
    f"Accuracy  : {rf_accuracy:.4f}"
)

print(
    f"Precision : {rf_precision:.4f}"
)

print(
    f"Recall    : {rf_recall:.4f}"
)

print(
    f"F1 Score  : {rf_f1:.4f}"
)


# ============================================================
# 17. MODEL COMPARISON
# ============================================================

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],

    "Accuracy": [
        lr_accuracy,
        rf_accuracy
    ],

    "Precision": [
        lr_precision,
        rf_precision
    ],

    "Recall": [
        lr_recall,
        rf_recall
    ],

    "F1 Score": [
        lr_f1,
        rf_f1
    ]
})


print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 18. SAVE RESULTS
# ============================================================

results.to_csv(
    "model_results.csv",
    index=False
)

print(
    "\nModel results saved as:"
    " model_results.csv"
)


# ============================================================
# 19. CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("RANDOM FOREST CLASSIFICATION REPORT")
print("==========================================")

report = classification_report(
    y_test,
    y_pred_rf,
    zero_division=0
)

print(report)

with open(
    "random_forest_classification_report.txt",
    "w"
) as file:

    file.write(report)


# ============================================================
# 20. CONFUSION MATRIX
# ============================================================

print("\nGenerating confusion matrix...")

labels = sorted(
    y_test.unique()
)

cm = confusion_matrix(
    y_test,
    y_pred_rf,
    labels=labels
)

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel(
    "Predicted Stress Level"
)

plt.ylabel(
    "Actual Stress Level"
)

plt.title(
    "Random Forest Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "random_forest_confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 21. STRESS LEVEL DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(7, 5)
)

sns.countplot(
    data=df,
    x="Stress_Level"
)

plt.xlabel(
    "Stress Level"
)

plt.ylabel(
    "Number of Students"
)

plt.title(
    "Student Stress-Level Distribution"
)

plt.tight_layout()

plt.savefig(
    "stress_level_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 22. EXAMPLE STUDENT PREDICTION
# ============================================================

print("\n==========================================")
print("EXAMPLE STUDENT STRESS PREDICTION")
print("==========================================")

sample_student = X_test.iloc[[0]]

actual_stress = y_test.iloc[0]

predicted_stress = (
    random_forest_model
    .predict(sample_student)[0]
)

print(
    "Actual Stress Level   :",
    actual_stress
)

print(
    "Predicted Stress Level:",
    predicted_stress
)


# ============================================================
# 23. SAVE SAMPLE DATA
# ============================================================

df.to_csv(
    "mongodb_stress_100k_sample.csv",
    index=False
)

print(
    "\nSample data saved as:"
    " mongodb_stress_100k_sample.csv"
)


# ============================================================
# 24. FINAL SUMMARY
# ============================================================

print("\n==========================================")
print("PROJECT EXECUTION COMPLETED")
print("==========================================")

print(
    "MongoDB documents available :",
    total_documents
)

print(
    "Documents used for ML       :",
    sample_size
)

print(
    "Training samples             :",
    len(X_train)
)

print(
    "Testing samples              :",
    len(X_test)
)

print("\nFinal Model Performance")

print(
    "Logistic Regression Accuracy:",
    round(lr_accuracy, 4)
)

print(
    "Random Forest Accuracy      :",
    round(rf_accuracy, 4)
)

print("\nGenerated files:")

print("1. model_results.csv")
print("2. random_forest_classification_report.txt")
print("3. random_forest_confusion_matrix.png")
print("4. stress_level_distribution.png")
print("5. mongodb_stress_100k_sample.csv")

print("\nMongoDB → ML pipeline completed successfully.")

