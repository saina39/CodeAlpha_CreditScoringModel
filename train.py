"""
train.py
--------
End-to-end training pipeline for the Credit Scoring Model.

What this script does:
1. Loads dataset/credit_data.csv
2. Cleans it (missing values, duplicates)
3. Encodes categorical variables + scales numeric ones
4. Runs EDA and saves plots to static/images/
5. Trains Logistic Regression, Decision Tree, Random Forest, XGBoost
6. Compares them on Accuracy / Precision / Recall / F1 / ROC-AUC
7. Saves the best model + scaler + encoders + feature list to model/

Run:
    python train.py
"""

import json
import os
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # no display needed, just save PNG files
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

DATASET_PATH = "dataset/credit_data.csv"
MODEL_DIR = "model"
IMAGES_DIR = "static/images"
TARGET_COL = "credit_risk"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


def load_data():
    print("Step 1: Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  Raw shape: {df.shape}")
    return df


def clean_data(df):
    print("Step 2: Cleaning data...")

    # Missing value report
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(f"  Missing values before cleaning:\n{missing.to_string() if len(missing) else '  None'}")

    # Numeric columns: fill missing with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL]
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Categorical columns: fill missing with mode
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  Removed {before - len(df)} duplicate rows")
    print(f"  Clean shape: {df.shape}")
    return df, numeric_cols, cat_cols


def run_eda(df, numeric_cols, cat_cols):
    print("Step 3: Running EDA and saving plots to static/images/...")

    # Class distribution
    plt.figure(figsize=(5, 4))
    df[TARGET_COL].map({0: "Low Risk", 1: "High Risk"}).value_counts().plot(
        kind="bar", color=["#4CAF50", "#F44336"]
    )
    plt.title("Class Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/class_distribution.png")
    plt.close()

    # Correlation heatmap (numeric features only)
    plt.figure(figsize=(9, 7))
    corr = df[numeric_cols + [TARGET_COL]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/correlation_heatmap.png")
    plt.close()

    # Distribution plots for numeric features
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="#2196F3")
        axes[i].set_title(col)
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/distributions.png")
    plt.close()

    print("  Saved: class_distribution.png, correlation_heatmap.png, distributions.png")


def encode_and_scale(df, numeric_cols, cat_cols):
    print("Step 4: Encoding categorical variables + scaling numeric features...")

    encoders = {}
    df_encoded = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le

    feature_cols = numeric_cols + cat_cols
    X = df_encoded[feature_cols]
    y = df_encoded[TARGET_COL]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    return X_scaled, y, encoders, scaler, feature_cols


def train_models(X_train, X_test, y_train, y_test):
    print("Step 5: Training and comparing models...")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, use_label_encoder=False,
            eval_metric="logloss", random_state=42
        )

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm,
        }
        trained_models[name] = model

        print(f"  {name}: Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f} | AUC={auc:.3f}")

    return results, trained_models


def plot_model_comparison(results):
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    model_names = list(results.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_names))
    width = 0.15

    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in model_names]
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/model_comparison.png")
    plt.close()
    print("  Saved: model_comparison.png")


def plot_feature_importance(model, feature_cols, model_name):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(
        [feature_cols[i] for i in order][::-1],
        [importances[i] for i in order][::-1],
        color="#3F51B5",
    )
    plt.title(f"Feature Importance ({model_name})")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/feature_importance.png")
    plt.close()
    print("  Saved: feature_importance.png")


def main():
    df = load_data()
    df, numeric_cols, cat_cols = clean_data(df)
    run_eda(df, numeric_cols, cat_cols)

    X, y, encoders, scaler, feature_cols = encode_and_scale(df, numeric_cols, cat_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results, trained_models = train_models(X_train, X_test, y_train, y_test)
    plot_model_comparison(results)

    # Pick the best model by ROC-AUC (a solid metric for imbalanced credit data)
    best_name = max(results, key=lambda m: results[m]["roc_auc"])
    best_model = trained_models[best_name]
    print(f"\nStep 6: Best model selected -> {best_name} (ROC-AUC={results[best_name]['roc_auc']})")

    plot_feature_importance(best_model, feature_cols, best_name)

    # Save everything predict.py / app.py will need
    joblib.dump(best_model, f"{MODEL_DIR}/credit_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(encoders, f"{MODEL_DIR}/encoders.pkl")

    metadata = {
        "best_model": best_name,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "results": results,
    }
    with open(f"{MODEL_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model/credit_model.pkl, scaler.pkl, encoders.pkl, metadata.json")
    print("Training complete.")


if __name__ == "__main__":
    main()
