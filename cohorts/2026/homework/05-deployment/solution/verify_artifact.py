"""Check the frozen artifacts against their recorded SHA-256 checksums.

Unpickling runs code, so we verify pipeline.bin before loading it.
"""

import hashlib
import json
import sys
from pathlib import Path

HOMEWORK_DIR = Path(__file__).resolve().parents[1]
EXPECTED_MODEL_SHA256 = "1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1"  # from homework.md


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(homework_dir: Path = HOMEWORK_DIR) -> bool:
    metadata = json.loads((homework_dir / "model_metadata.json").read_text())
    checks = {
        "pipeline.bin vs homework.md": (sha256(homework_dir / "pipeline.bin"), EXPECTED_MODEL_SHA256),
        "pipeline.bin vs metadata": (sha256(homework_dir / "pipeline.bin"), metadata["model_sha256"]),
        "feature_defaults.json vs metadata": (
            sha256(homework_dir / "feature_defaults.json"),
            metadata["feature_defaults_sha256"],
        ),
    }
    ok = True
    for name, (actual, expected) in checks.items():
        match = actual == expected
        ok &= match
        print(f"{'OK ' if match else 'BAD'} {name}: {actual[:16]}…")
    return ok


if __name__ == "__main__":
    sys.exit(0 if verify() else 1)
