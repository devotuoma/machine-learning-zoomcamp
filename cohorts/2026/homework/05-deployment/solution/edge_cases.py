"""Probe the live API with inputs a real client might send.

The service accepts every one of these with HTTP 200 unless pydantic rejects
the type. DictVectorizer silently drops category values it never saw in
training, so a typo becomes an all-zero one-hot block.

Usage: python solution/edge_cases.py [base_url] [--save results/edge_cases.json]
"""

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

SOLUTION_DIR = Path(__file__).resolve().parent
BASE = json.loads((SOLUTION_DIR / "leads" / "q3_lead.json").read_text())

CASES = {
    "baseline (Q3 lead)": {},
    "lead_source='Paid_Ads' (capitalised)": {"lead_source": "Paid_Ads"},
    "lead_source='tiktok' (never seen)": {"lead_source": "tiktok"},
    "lead_source=null (missing → 'NA')": {"lead_source": None},
    "location='North America' (space, caps)": {"location": "North America"},
    "annual_income=null (→ median 55051)": {"annual_income": None},
    "annual_income='79276' (string)": {"annual_income": "79276"},
    "annual_income=7927600 (typo ×100)": {"annual_income": 7927600.0},
    "number_of_courses_viewed=2.0": {"number_of_courses_viewed": 2.0},
    "number_of_courses_viewed=2.5": {"number_of_courses_viewed": 2.5},
    "interaction_count=-3 (negative)": {"interaction_count": -3},
    "empty body {}": None,
}


def post(url: str, payload: dict) -> tuple[int, dict]:
    req = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=10) as r:
            return r.status, json.load(r)
    except HTTPError as e:
        return e.code, json.load(e)


def run(base_url: str) -> list[dict]:
    rows = []
    for name, change in CASES.items():
        payload = {} if change is None else {**BASE, **change}
        status, body = post(f"{base_url}/predict", payload)
        p = body.get("conversion_probability")
        rows.append({"case": name, "status": status, "probability": p, "payload_change": change})
        shown = f"{p:.3f}" if p is not None else body["detail"][0]["msg"][:45]
        print(f"{status}  {name:<42} {shown}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url", nargs="?", default="http://localhost:9696")
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()
    rows = run(args.base_url)
    if args.save:
        args.save.write_text(json.dumps(rows, indent=2) + "\n")
