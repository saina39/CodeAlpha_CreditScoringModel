"""
generate_dataset.py
--------------------
Generates a realistic, German-Credit-style synthetic dataset and saves it as
dataset/credit_data.csv

Why this exists:
UCI/Kaggle dataset downloads require internet access to external hosts that
may not always be reachable from every machine. This script creates a
statistically realistic substitute with the SAME feature set and logic as
the classic German Credit dataset, so train.py always has data to work with.

If you want to use the REAL UCI German Credit dataset instead, download it
from:
  https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
and place it as dataset/credit_data.csv with matching column names, or adjust
the column names in train.py.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 1500  # number of synthetic applicants

# ---- Core numeric features ----
age = np.random.randint(19, 70, N)
income = np.random.normal(45000, 18000, N).clip(8000, 200000).round(0)
employment_length = np.random.randint(0, 40, N)
debt = (income * np.random.uniform(0.05, 0.9, N)).round(0)
credit_history_length = np.random.randint(0, 30, N)
num_credit_cards = np.random.randint(0, 8, N)
num_loans = np.random.randint(0, 5, N)
monthly_emi = (debt / (np.random.randint(6, 60, N))).round(0)
existing_defaults = np.random.choice([0, 1, 2, 3], N, p=[0.75, 0.15, 0.07, 0.03])

# ---- Categorical features ----
payment_history = np.random.choice(
    ["excellent", "good", "average", "poor"], N, p=[0.3, 0.35, 0.25, 0.10]
)
housing = np.random.choice(["own", "rent", "with_parents"], N, p=[0.5, 0.4, 0.1])
purpose = np.random.choice(
    ["car", "education", "business", "home_renovation", "personal", "medical"], N
)
job_type = np.random.choice(
    ["salaried", "self_employed", "unemployed", "student"], N, p=[0.6, 0.2, 0.1, 0.1]
)

# ---- Build a "risk score" using domain logic, then derive the target ----
payment_map = {"excellent": 0, "good": 1, "average": 2, "poor": 3}
job_map = {"salaried": 0, "self_employed": 1, "student": 2, "unemployed": 3}

risk_score = (
    (debt / (income + 1)) * 3.0
    + existing_defaults * 1.5
    + pd.Series(payment_history).map(payment_map).values * 1.2
    + pd.Series(job_type).map(job_map).values * 0.8
    + (monthly_emi / (income / 12 + 1)) * 2.5
    - (credit_history_length * 0.05)
    - (employment_length * 0.03)
    - (num_credit_cards.clip(max=3) * 0.1)  # having some credit history is fine
    + np.random.normal(0, 0.6, N)  # noise
)

# Threshold risk_score into a binary target: 1 = High Risk, 0 = Low Risk
threshold = np.percentile(risk_score, 70)  # ~30% high risk, realistic class imbalance
target = (risk_score > threshold).astype(int)

df = pd.DataFrame(
    {
        "age": age,
        "income": income,
        "employment_length": employment_length,
        "debt": debt,
        "credit_history_length": credit_history_length,
        "num_credit_cards": num_credit_cards,
        "num_loans": num_loans,
        "monthly_emi": monthly_emi,
        "payment_history": payment_history,
        "housing": housing,
        "purpose": purpose,
        "job_type": job_type,
        "existing_defaults": existing_defaults,
        "credit_risk": target,  # 0 = Low Risk, 1 = High Risk
    }
)

# Inject a few missing values and duplicate rows so train.py's cleaning steps
# have something real to do (mirrors real-world messy data)
missing_idx = np.random.choice(df.index, size=30, replace=False)
df.loc[missing_idx, "income"] = np.nan
missing_idx2 = np.random.choice(df.index, size=15, replace=False)
df.loc[missing_idx2, "employment_length"] = np.nan

df = pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)  # duplicates

df.to_csv("dataset/credit_data.csv", index=False)
print(f"Saved dataset/credit_data.csv with shape {df.shape}")
print(df["credit_risk"].value_counts(normalize=True))
