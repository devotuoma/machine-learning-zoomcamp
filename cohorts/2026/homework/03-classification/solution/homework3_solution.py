"""ML Zoomcamp 2026 - Homework 3: Machine Learning for Classification.

Predicts `converted` (did the lead sign up?) on the pinned 2026 lead-scoring
dataset with logistic regression, following the module 03 lessons.

Usage:
    python homework3_solution.py [path/to/course_lead_scoring_2026.csv]
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.model_selection import train_test_split

DATA_URL = (
    "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/"
    "main/cohorts/2026/data/course_lead_scoring_2026.csv"
)
LOCAL_DATA = Path(__file__).resolve().parents[3] / "data" / "course_lead_scoring_2026.csv"

TARGET = "converted"
CATEGORICAL = ["lead_source", "industry", "employment_status", "location"]
NUMERICAL = ["annual_income", "number_of_courses_viewed", "interaction_count", "lead_score"]


# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------
def load_data(path=None):
    if path is None:
        path = LOCAL_DATA if LOCAL_DATA.exists() else DATA_URL
    df = pd.read_csv(path)

    print("Missing values per column:")
    print(df.isnull().sum().to_string(), "\n")

    df[CATEGORICAL] = df[CATEGORICAL].fillna("NA")
    df[NUMERICAL] = df[NUMERICAL].fillna(0.0)
    return df


def split(df):
    """60/20/20 split with the exact calls from the homework."""
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train.pop(TARGET).values
    y_val = df_val.pop(TARGET).values
    y_test = df_test.pop(TARGET).values
    return df_train, df_val, df_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def train_and_score(df_train, y_train, df_val, y_val, features, C=1.0):
    """One-hot encode with DictVectorizer, fit logistic regression, return val accuracy."""
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train[features].to_dict(orient="records"))
    X_val = dv.transform(df_val[features].to_dict(orient="records"))

    model = LogisticRegression(solver="liblinear", C=C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return accuracy_score(y_val, model.predict(X_val))


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
def main(path=None):
    df = load_data(path)
    features = CATEGORICAL + NUMERICAL

    # Q1
    print("Q1. industry value counts:")
    print(df.industry.value_counts().to_string())
    print(f"-> mode = {df.industry.mode()[0]}\n")

    # Q2
    corr = df[NUMERICAL].corr()
    print("Q2. Correlation matrix (numerical features):")
    print(corr.round(4).to_string())
    pairs = [
        ("interaction_count", "lead_score"),
        ("number_of_courses_viewed", "lead_score"),
        ("number_of_courses_viewed", "interaction_count"),
        ("annual_income", "interaction_count"),
    ]
    pair_corr = {f"{a} & {b}": corr.loc[a, b] for a, b in pairs}
    for k, v in pair_corr.items():
        print(f"   {k}: {v:.4f}")
    print(f"-> biggest |r|: {max(pair_corr, key=lambda k: abs(pair_corr[k]))}\n")

    df_train, df_val, df_test, y_train, y_val, y_test = split(df)
    print(f"Split sizes: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}\n")

    # Q3
    print("Q3. Mutual information with `converted` (train only):")
    mi = {c: round(mutual_info_score(y_train, df_train[c]), 2) for c in CATEGORICAL}
    for k, v in sorted(mi.items(), key=lambda kv: -kv[1]):
        print(f"   {k}: {v}")
    print(f"-> biggest MI: {max(mi, key=mi.get)}\n")

    # Q4
    base_acc = train_and_score(df_train, y_train, df_val, y_val, features)
    print(f"Q4. Validation accuracy (C=1.0): {base_acc:.4f} -> {round(base_acc, 2)}\n")

    # Q5
    print("Q5. Feature elimination (difference = base_acc - acc_without):")
    diffs = {}
    for f in features:
        subset = [x for x in features if x != f]
        acc = train_and_score(df_train, y_train, df_val, y_val, subset)
        diffs[f] = base_acc - acc
        print(f"   without {f:<26} acc={acc:.4f}  diff={diffs[f]:+.4f}")
    candidates = ["lead_source", "number_of_courses_viewed", "interaction_count"]
    print(f"-> smallest diff among options: {min(candidates, key=lambda f: diffs[f])}\n")

    # Q6
    print("Q6. Regularization sweep:")
    scores = {}
    for C in [0.000001, 0.00001, 0.0001, 0.001]:
        scores[C] = round(train_and_score(df_train, y_train, df_val, y_val, features, C=C), 3)
        print(f"   C={C:<8} acc={scores[C]}")
    best = max(scores.values())
    print(f"-> best C (smallest on ties): {min(c for c, s in scores.items() if s == best)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
