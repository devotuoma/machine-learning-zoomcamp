import json

import pytest

from q4_client import check, post_json


def test_q4_api_probability(api_url, lead_q4):
    out = check(api_url, lead_q4)
    assert out["prediction"]["conversion_probability"] == pytest.approx(0.770, abs=0.0005)
    assert out["prediction"]["conversion"] is True


def test_api_matches_offline_model(api_url, lead_q3):
    from q3_load_model import conversion_probability, load_pipeline

    served = post_json(f"{api_url}/predict", lead_q3)["conversion_probability"]
    assert served == pytest.approx(conversion_probability(load_pipeline(), lead_q3), abs=1e-6)


def test_health_reports_metadata_checksum(api_url):
    from q4_client import HOMEWORK_DIR, get_json

    meta = json.loads((HOMEWORK_DIR / "model_metadata.json").read_text())
    health = get_json(f"{api_url}/health")
    assert health["model_sha256"] == meta["model_sha256"]
    assert health["feature_defaults_sha256"] == meta["feature_defaults_sha256"]
