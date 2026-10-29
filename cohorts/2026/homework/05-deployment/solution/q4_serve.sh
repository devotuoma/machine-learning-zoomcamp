#!/usr/bin/env bash
# Question 4: serve the frozen model with the provided FastAPI app.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --locked
exec uv run --locked uvicorn predict:app --host 0.0.0.0 --port "${PORT:-9696}"
