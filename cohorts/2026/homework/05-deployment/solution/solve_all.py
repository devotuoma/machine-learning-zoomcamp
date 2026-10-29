"""Collect all Homework 5 answers into results/answers.json.

Q4/Q6 are read from the saved service responses, so start the API
(and the container) first if those files are missing.

Usage (from 05-deployment): uv run --locked python solution/solve_all.py
"""

import json
import subprocess
import sys
from pathlib import Path

SOLUTION_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SOLUTION_DIR.parent), str(SOLUTION_DIR)]

from q2_locked_version import locked_version  # noqa: E402
from q3_load_model import conversion_probability, load_pipeline  # noqa: E402
from q5_dockerfile_base import base_image  # noqa: E402
from verify_artifact import verify  # noqa: E402

RESULTS = SOLUTION_DIR / "results"


def saved_probability(name: str):
    path = RESULTS / name
    if not path.exists():
        return None
    return round(json.loads(path.read_text())["prediction"]["conversion_probability"], 3)


def main() -> dict:
    assert verify(), "artifact checksum mismatch"
    q3_lead = json.loads((SOLUTION_DIR / "leads" / "q3_lead.json").read_text())
    base = base_image()
    answers = {
        "q1_uv_version": subprocess.run(["uv", "--version"], capture_output=True, text=True).stdout.split()[1],
        "q2_scikit_learn": locked_version("scikit-learn"),
        "q3_probability": round(conversion_probability(load_pipeline(), q3_lead), 3),
        "q4_probability": saved_probability("q4_api_response.json"),
        "q5_base_image": f"{base['image']}:{base['tag']}",
        "q6_probability": saved_probability("q6_container_response.json"),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "answers.json").write_text(json.dumps(answers, indent=2) + "\n")
    return answers


if __name__ == "__main__":
    for k, v in main().items():
        print(f"{k:<18} {v}")
