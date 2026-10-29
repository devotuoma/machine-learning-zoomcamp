"""Question 5: which Python base image does the Dockerfile declare?

Parses the first FROM line, splitting the tag from the pinned digest.
"""

import re
from pathlib import Path

HOMEWORK_DIR = Path(__file__).resolve().parents[1]
FROM_RE = re.compile(r"^FROM\s+(?P<image>[^\s@:]+):(?P<tag>[^\s@]+)(?:@(?P<digest>\S+))?", re.MULTILINE)


def base_image(dockerfile: Path = HOMEWORK_DIR / "Dockerfile") -> dict:
    match = FROM_RE.search(dockerfile.read_text())
    if not match:
        raise ValueError("no FROM line found")
    return match.groupdict()


if __name__ == "__main__":
    base = base_image()
    print(f"image:  {base['image']}")
    print(f"tag:    {base['tag']}")
    print(f"digest: {base['digest']}")
    print(f"\nQ5 answer: {base['image']}:{base['tag']}")
