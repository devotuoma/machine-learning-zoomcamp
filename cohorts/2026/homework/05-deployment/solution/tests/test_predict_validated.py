import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import pytest

from conftest import HOMEWORK_DIR, SOLUTION_DIR
from edge_cases import BASE, post


@pytest.fixture(scope="module")
def strict_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "--app-dir", str(SOLUTION_DIR), "predict_validated:app",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=HOMEWORK_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            urlopen(f"{url}/docs", timeout=1)
            break
        except OSError:
            time.sleep(0.25)
    yield url
    proc.terminate()
    proc.wait(timeout=10)


def test_same_probability_as_course_api(strict_url):
    status, body = post(f"{strict_url}/predict", BASE)
    assert status == 200
    assert body["conversion_probability"] == pytest.approx(0.533, abs=0.0005)
    assert body["warnings"] == []


@pytest.mark.parametrize("field,value", [("lead_source", "Paid_Ads"), ("location", "North America")])
def test_harmless_typos_are_normalised(strict_url, field, value):
    _, body = post(f"{strict_url}/predict", {**BASE, field: value})
    assert body["conversion_probability"] == pytest.approx(0.533, abs=0.0005)


@pytest.mark.parametrize("change", [
    {"lead_source": "tiktok"},
    {"interaction_count": -3},
    {"annual_income": 7927600.0},
    {"lead_score": 4.1},
    {"leadscore": 0.4},
])
def test_bad_values_are_rejected(strict_url, change):
    status, _ = post(f"{strict_url}/predict", {**BASE, **change})
    assert status == 422


def test_missing_values_still_use_training_defaults(strict_url):
    status, body = post(f"{strict_url}/predict", {})
    assert status == 200
    assert body["conversion_probability"] == pytest.approx(0.610, abs=0.0005)


def test_out_of_range_income_warns(strict_url):
    _, body = post(f"{strict_url}/predict", {**BASE, "annual_income": 250000.0})
    assert body["warnings"] and "annual_income" in body["warnings"][0]
