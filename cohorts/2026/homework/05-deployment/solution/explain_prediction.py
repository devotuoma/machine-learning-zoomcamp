"""Explain a prediction: break the logistic regression's log-odds into
one contribution per feature (weight × value), plus the intercept.

Usage (from the 05-deployment directory):
    uv run --with matplotlib python solution/explain_prediction.py [lead.json] [--plot out.png]
"""

import argparse
import json
import math
import sys
from pathlib import Path

SOLUTION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOLUTION_DIR.parent))

from model import normalize_record  # noqa: E402
from q3_load_model import load_pipeline  # noqa: E402


def contributions(pipeline, lead: dict) -> tuple[float, dict[str, float]]:
    dv, lr = pipeline.steps[0][1], pipeline.steps[-1][1]
    x = dv.transform([normalize_record(lead)]).toarray()[0]
    names = dv.get_feature_names_out()
    parts = {n: w * v for n, w, v in zip(names, lr.coef_[0], x) if v != 0}
    return float(lr.intercept_[0]), parts


def plot(intercept: float, parts: dict[str, float], out: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    items = sorted(parts.items(), key=lambda kv: abs(kv[1]))
    labels = ["intercept"] + [k for k, _ in items]
    values = [intercept] + [v for _, v in items]
    colors = ["#9aa5b1"] + ["#1a9e5f" if v > 0 else "#d64545" for _, v in items]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.barh(labels, values, color=colors)
    ax.bar_label(bars, fmt="%+.3f", padding=3, fontsize=8)
    ax.axvline(0, c="black", lw=0.8)
    ax.set_xlabel("contribution to log-odds")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(min(values) - 0.35, max(values) + 0.35)
    plt.tight_layout()
    plt.savefig(out, dpi=110)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lead", nargs="?", type=Path, default=SOLUTION_DIR / "leads" / "q3_lead.json")
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    pipeline = load_pipeline()
    intercept, parts = contributions(pipeline, json.loads(args.lead.read_text()))
    logit = intercept + sum(parts.values())
    p = 1 / (1 + math.exp(-logit))

    print(f"{'intercept':<34}{intercept:+.4f}")
    for name, value in sorted(parts.items(), key=lambda kv: -abs(kv[1])):
        print(f"{name:<34}{value:+.4f}")
    print(f"{'= log-odds':<34}{logit:+.4f}")
    print(f"sigmoid → P(converted) = {p:.6f}")

    if args.plot:
        plot(intercept, parts, args.plot, f"{args.lead.stem}: P(converted) = {p:.3f}")
