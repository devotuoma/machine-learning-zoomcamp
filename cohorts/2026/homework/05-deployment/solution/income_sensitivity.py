"""What-if: how does P(converted) for the Q3 lead change with annual_income?

The model was trained on unscaled income, so its tiny per-dollar weight
(about -1.2e-5) adds up: every extra 10,000 removes about 0.12 from the log-odds.

Usage: uv run --with matplotlib python solution/income_sensitivity.py [--plot out.png]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SOLUTION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOLUTION_DIR.parent))

from q3_load_model import conversion_probability, load_pipeline  # noqa: E402

INCOMES = np.arange(0, 200_001, 2_500)


def sweep(pipeline, lead: dict, incomes=INCOMES) -> np.ndarray:
    return np.array([conversion_probability(pipeline, {**lead, "annual_income": float(i)}) for i in incomes])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    lead = json.loads((SOLUTION_DIR / "leads" / "q3_lead.json").read_text())
    probs = sweep(load_pipeline(), lead)
    for i in [0, 25_000, 55_051, 79_276, 100_000, 150_000, 200_000]:
        print(f"income {i:>7,}: P = {conversion_probability(load_pipeline(), {**lead, 'annual_income': float(i)}):.3f}")
    flip = INCOMES[np.argmax(probs < 0.5)]
    print(f"decision flips to 'not converted' at about {flip:,}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.plot(INCOMES / 1000, probs, c="#2f6fdf", lw=2)
        ax.axhline(0.5, ls="--", c="grey", lw=1)
        ax.axvline(lead["annual_income"] / 1000, c="#e8833a", lw=1)
        ax.annotate(f"Q3 lead\n{lead['annual_income']/1000:.0f}k → {conversion_probability(load_pipeline(), lead):.3f}",
                    (lead["annual_income"] / 1000, 0.533), xytext=(110, 0.72), arrowprops=dict(arrowstyle="->"))
        ax.annotate(f"flips at ~{flip/1000:.0f}k", (flip / 1000, 0.5), xytext=(flip / 1000 + 25, 0.38),
                    arrowprops=dict(arrowstyle="->"))
        ax.set_xlabel("annual_income (thousands)")
        ax.set_ylabel("P(converted)")
        ax.set_ylim(0, 1)
        ax.set_title("Same lead, different income: the unscaled feature decides")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=110)
