"""Question 3: load pipeline.bin with pickle and score a lead.

Usage (from the 05-deployment directory):
    uv run python solution/q3_load_model.py [path/to/lead.json]
"""

import json
import pickle
import sys
from pathlib import Path

SOLUTION_DIR = Path(__file__).resolve().parent
HOMEWORK_DIR = SOLUTION_DIR.parent
sys.path.insert(0, str(HOMEWORK_DIR))

from model import normalize_record  # noqa: E402  (same preprocessing as the API)
from verify_artifact import verify  # noqa: E402


def load_pipeline(path: Path = HOMEWORK_DIR / "pipeline.bin"):
    with path.open("rb") as f:
        return pickle.load(f)


def conversion_probability(pipeline, lead: dict) -> float:
    record = normalize_record(lead)
    return float(pipeline.predict_proba([record])[0, 1])


if __name__ == "__main__":
    lead_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SOLUTION_DIR / "leads" / "q3_lead.json"
    if not verify():
        sys.exit("Artifact checksum mismatch: refusing to unpickle.")

    lead = json.loads(lead_path.read_text())
    p = conversion_probability(load_pipeline(), lead)
    print(f"\nconversion probability: {p:.6f}")
    print(f"Q3 answer: {p:.3f}")
