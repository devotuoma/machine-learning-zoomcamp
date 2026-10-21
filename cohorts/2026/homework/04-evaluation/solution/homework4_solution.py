"""ML Zoomcamp 2026 - Homework 4: Evaluation Metrics for Classification.

Evaluates a logistic regression model that predicts `converted` (did the lead
sign up?) on the pinned 2026 lead-scoring dataset: ROC AUC, precision/recall,
F1 and K-fold cross-validation, following the module 04 lessons.

Usage:
    python homework4_solution.py [path/to/course_lead_scoring_2026.csv]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

DATA_URL = (
    "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/"
    "main/cohorts/2026/data/course_lead_scoring_2026.csv"
)
LOCAL_DATA = Path(__file__).resolve().parents[3] / "data" / "course_lead_scoring_2026.csv"

TARGET = "converted"
CATEGORICAL = ["lead_source", "industry", "employment_status", "location"]
NUMERICAL = ["number_of_courses_viewed", "annual_income", "interaction_count", "lead_score"]
FEATURES = CATEGORICAL + NUMERICAL
THRESHOLDS = np.linspace(0, 1, 101)


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
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

    df_full_train = df_full_train.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train.pop(TARGET).values
    y_val = df_val.pop(TARGET).values
    y_test = df_test.pop(TARGET).values
    return df_full_train, df_train, df_val, df_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def train(df, y, C=1.0):
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(df[FEATURES].to_dict(orient="records"))
    model = LogisticRegression(solver="liblinear", C=C, max_iter=1000)
    model.fit(X, y)
    return dv, model


def predict(df, dv, model):
    X = dv.transform(df[FEATURES].to_dict(orient="records"))
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def precision_recall_f1(y_true, y_pred, thresholds=THRESHOLDS):
    """Confusion-matrix based precision, recall and F1 for every threshold."""
    rows = []
    actual_pos = y_true == 1
    actual_neg = ~actual_pos
    for t in thresholds:
        predict_pos = y_pred >= t
        tp = (predict_pos & actual_pos).sum()
        fp = (predict_pos & actual_neg).sum()
        fn = (~predict_pos & actual_pos).sum()

        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        rows.append((round(t, 2), tp, fp, fn, p, r, f1))
    return pd.DataFrame(rows, columns=["threshold", "tp", "fp", "fn", "precision", "recall", "f1"])


def cross_validate(df_full_train, C=1.0, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    scores = []
    for train_idx, val_idx in kfold.split(df_full_train):
        df_tr = df_full_train.iloc[train_idx]
        df_va = df_full_train.iloc[val_idx]
        dv, model = train(df_tr, df_tr[TARGET].values, C=C)
        scores.append(roc_auc_score(df_va[TARGET].values, predict(df_va, dv, model)))
    return np.array(scores)


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
def main(path=None):
    df = load_data(path)
    df_full_train, df_train, df_val, df_test, y_train, y_val, y_test = split(df)
    print(f"Split sizes: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}\n")

    # Q1
    print("Q1. ROC AUC of each numerical feature as a score (train only):")
    feature_auc = {}
    for c in NUMERICAL:
        auc = roc_auc_score(y_train, df_train[c])
        if auc < 0.5:
            auc = roc_auc_score(y_train, -df_train[c])
            print(f"   {c:<26} inverted")
        feature_auc[c] = auc
    for c, auc in sorted(feature_auc.items(), key=lambda kv: -kv[1]):
        print(f"   {c:<26} AUC={auc:.4f}")
    print(f"-> highest AUC: {max(feature_auc, key=feature_auc.get)}\n")

    # Q2
    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)
    val_auc = roc_auc_score(y_val, y_pred)
    print(f"Q2. Validation AUC (C=1.0): {val_auc:.4f} -> {round(val_auc, 3)}\n")

    # Q3
    metrics = precision_recall_f1(y_val, y_pred)
    nonzero = metrics[(metrics.precision > 0) | (metrics.recall > 0)].copy()
    nonzero["diff"] = (nonzero.precision - nonzero.recall).abs()
    best_pr = nonzero.loc[nonzero["diff"].idxmin()]
    print("Q3. Precision vs recall around the intersection:")
    print(nonzero.nsmallest(5, "diff")[["threshold", "precision", "recall", "diff"]]
          .round(4).to_string(index=False))
    print(f"-> intersection at threshold {best_pr.threshold:.2f} "
          f"(P={best_pr.precision:.4f}, R={best_pr.recall:.4f})\n")

    # Q4
    best_f1 = metrics.loc[metrics.f1.idxmax()]
    print("Q4. Top F1 thresholds:")
    print(metrics.nlargest(5, "f1")[["threshold", "precision", "recall", "f1"]]
          .round(4).to_string(index=False))
    print(f"-> max F1 = {best_f1.f1:.4f} at threshold {best_f1.threshold:.2f}\n")

    # Q5
    scores = cross_validate(df_full_train)
    print("Q5. 5-fold CV AUC (C=1.0):", np.round(scores, 4))
    print(f"-> mean={scores.mean():.4f}, std={scores.std():.4f} -> {round(scores.std(), 3)}\n")

    # Q6
    print("Q6. 5-fold CV for different C:")
    results = []
    for C in [0.000001, 0.001, 1]:
        s = cross_validate(df_full_train, C=C)
        mean, std = round(s.mean(), 3), round(s.std(), 3)
        results.append((C, mean, std))
        print(f"   C={C:<8} mean={mean:.3f}  std={std:.3f}")
    # best mean, then lowest std, then smallest C
    best_C = sorted(results, key=lambda r: (-r[1], r[2], r[0]))[0][0]
    print(f"-> best C: {best_C}\n")

    return {
        "feature_auc": feature_auc,
        "val_auc": val_auc,
        "metrics": metrics,
        "pr_threshold": best_pr.threshold,
        "f1_threshold": best_f1.threshold,
        "cv_scores": scores,
        "c_results": results,
        "best_C": best_C,
    }


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
