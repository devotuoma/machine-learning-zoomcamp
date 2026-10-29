"""Questions 4 and 6: call /health and /predict on a running service.

Usage:
    python solution/q4_client.py [base_url] [path/to/lead.json] [--save results/x.json]

Standard library only, so it runs outside the uv environment too.
"""

import argparse
import json
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

SOLUTION_DIR = Path(__file__).resolve().parent
HOMEWORK_DIR = SOLUTION_DIR.parent


def get_json(url: str) -> dict:
    with urlopen(url, timeout=10) as r:
        return json.load(r)


def post_json(url: str, payload: dict) -> dict:
    req = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=10) as r:
        return json.load(r)


def wait_until_healthy(base_url: str, timeout: float = 30) -> dict:
    deadline = time.time() + timeout
    while True:
        try:
            return get_json(f"{base_url}/health")
        except (URLError, ConnectionError):
            if time.time() > deadline:
                raise
            time.sleep(0.5)


def check(base_url: str, lead: dict) -> dict:
    health = wait_until_healthy(base_url)
    expected = json.loads((HOMEWORK_DIR / "model_metadata.json").read_text())["model_sha256"]
    assert health["status"] == "ok", health
    assert health["model_sha256"] == expected, "service is running a different pipeline.bin"

    result = post_json(f"{base_url}/predict", lead)
    assert set(result) == {"conversion_probability", "conversion"}, result
    assert 0 <= result["conversion_probability"] <= 1
    assert isinstance(result["conversion"], bool)
    return {"health": health, "prediction": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url", nargs="?", default="http://localhost:9696")
    parser.add_argument("lead", nargs="?", type=Path, default=SOLUTION_DIR / "leads" / "q4_lead.json")
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()

    out = check(args.base_url, json.loads(args.lead.read_text()))
    print("health:    ", out["health"])
    print("prediction:", out["prediction"])
    print(f"answer:     {out['prediction']['conversion_probability']:.3f}")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(out, indent=2) + "\n")
