import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import pytest

SOLUTION_DIR = Path(__file__).resolve().parents[1]
HOMEWORK_DIR = SOLUTION_DIR.parent
sys.path[:0] = [str(HOMEWORK_DIR), str(SOLUTION_DIR)]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def lead_q3() -> dict:
    return json.loads((SOLUTION_DIR / "leads" / "q3_lead.json").read_text())


@pytest.fixture(scope="session")
def lead_q4() -> dict:
    return json.loads((SOLUTION_DIR / "leads" / "q4_lead.json").read_text())


@pytest.fixture(scope="session")
def api_url():
    """Start the real uvicorn app on a free port for the whole session."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "predict:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=HOMEWORK_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            urlopen(f"{url}/health", timeout=1)
            break
        except OSError:
            time.sleep(0.25)
    else:
        proc.kill()
        pytest.fail("uvicorn did not start")
    yield url
    proc.terminate()
    proc.wait(timeout=10)
