"""Question 2: which scikit-learn version does the 2026 reference environment lock?

Reads uv.lock (TOML) instead of the installed package, so the answer comes
from the version-controlled file the grader uses.
"""

import tomllib
from pathlib import Path

HOMEWORK_DIR = Path(__file__).resolve().parents[1]


def locked_version(package: str, lock_path: Path = HOMEWORK_DIR / "uv.lock") -> str:
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    for entry in lock["package"]:
        if entry["name"] == package:
            return entry["version"]
    raise KeyError(f"{package} not found in {lock_path}")


def pinned_version(package: str, pyproject_path: Path = HOMEWORK_DIR / "pyproject.toml") -> str:
    project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]
    for spec in project["dependencies"]:
        name, _, version = spec.partition("==")
        if name == package:
            return version
    raise KeyError(f"{package} not pinned in {pyproject_path}")


if __name__ == "__main__":
    for pkg in ["scikit-learn", "numpy", "fastapi", "uvicorn"]:
        print(f"{pkg:<14} pyproject={pinned_version(pkg):<8} uv.lock={locked_version(pkg)}")
    print(f"\nQ2 answer: scikit-learn {locked_version('scikit-learn')}")
