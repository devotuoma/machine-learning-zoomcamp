"""ML Zoomcamp 2026 - Homework 2: Machine Learning for Regression.

Predicts `fuel_efficiency_mpg` with linear regression implemented from scratch
via the normal equation (no scikit-learn), exactly as in module 02 lessons.

Usage:
    python homework2_solution.py [path/to/car_fuel_efficiency_2026.csv]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/"
    "main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
)
LOCAL_DATA = Path(__file__).resolve().parents[3] / "data" / "car_fuel_efficiency_2026.csv"

BASE = ["engine_displacement", "horsepower", "vehicle_weight", "model_year"]
TARGET = "fuel_efficiency_mpg"


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------
def load_data(path=None):
    if path is None:
        path = LOCAL_DATA if LOCAL_DATA.exists() else DATA_URL
    df = pd.read_csv(path)
    return df[BASE + [TARGET]]


def split(df, seed):
    """60/20/20 shuffle split, identical to the lecture code."""
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
    df_val = df.iloc[idx[n_train:n_train + n_val]].reset_index(drop=True)
    df_test = df.iloc[idx[n_train + n_val:]].reset_index(drop=True)
    return df_train, df_val, df_test


def xy(df, fill_value=0):
    X = df[BASE].fillna(fill_value).values
    y = df[TARGET].values
    return X, y


# ---------------------------------------------------------------------------
# Model (normal equation) & metric
# ---------------------------------------------------------------------------
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    return w_full[0], w_full[1:]


def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    return w_full[0], w_full[1:]


def predict(X, w0, w):
    return w0 + X.dot(w)


def rmse(y, y_pred):
    return np.sqrt(((y - y_pred) ** 2).mean())


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
def main(path=None):
    df = load_data(path)
    print(f"Dataset shape: {df.shape}")

    # EDA: long tail?
    t = df[TARGET]
    print(f"\nEDA {TARGET}: mean={t.mean():.3f} median={t.median():.3f} "
          f"std={t.std():.3f} skew={t.skew():.3f} min={t.min()} max={t.max()}")

    # Q1
    missing = df.isnull().sum()
    print("\nQ1 missing values per column:\n" + missing.to_string())
    q1 = missing[missing > 0].index.tolist()
    print(f"Q1 answer: {q1}")

    # Q2
    q2 = df["horsepower"].median()
    print(f"\nQ2 median horsepower: {q2}")

    # Q3
    df_train, df_val, _ = split(df, 42)
    y_train, y_val = df_train[TARGET].values, df_val[TARGET].values
    mean_hp = df_train["horsepower"].mean()  # training statistics only

    q3 = {}
    for name, fill in [("With 0", 0), ("With mean", mean_hp)]:
        X_train, _ = xy(df_train, fill)
        X_val, _ = xy(df_val, fill)
        w0, w = train_linear_regression(X_train, y_train)
        q3[name] = round(rmse(y_val, predict(X_val, w0, w)), 3)
    print(f"\nQ3 train mean horsepower = {mean_hp:.4f}")
    print(f"Q3 RMSE: {q3}")

    # Q4
    X_train, _ = xy(df_train, 0)
    X_val, _ = xy(df_val, 0)
    q4 = {}
    for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
        w0, w = train_linear_regression_reg(X_train, y_train, r=r)
        q4[r] = round(rmse(y_val, predict(X_val, w0, w)), 4)
    best = min(q4.values())
    best_r = min(r for r, s in q4.items() if s == best)
    print(f"\nQ4 RMSE by r: {q4}")
    print(f"Q4 best r: {best_r}")

    # Q5
    scores = []
    for seed in range(10):
        tr, va, _ = split(df, seed)
        X_tr, y_tr = xy(tr, 0)
        X_va, y_va = xy(va, 0)
        w0, w = train_linear_regression(X_tr, y_tr)
        scores.append(rmse(y_va, predict(X_va, w0, w)))
    print(f"\nQ5 RMSE per seed: {[round(s, 4) for s in scores]}")
    print(f"Q5 std: {round(np.std(scores), 3)}")

    # Q6
    tr, va, te = split(df, 9)
    full = pd.concat([tr, va]).reset_index(drop=True)
    X_full, y_full = xy(full, 0)
    X_te, y_te = xy(te, 0)
    w0, w = train_linear_regression_reg(X_full, y_full, r=0.001)
    q6 = rmse(y_te, predict(X_te, w0, w))
    print(f"\nQ6 test RMSE (seed 9, r=0.001): {q6:.4f}  (rounded 3dp: {round(q6, 3)})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
