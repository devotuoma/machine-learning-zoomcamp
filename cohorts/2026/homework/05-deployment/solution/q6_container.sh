#!/usr/bin/env bash
# Question 6: build the canonical image, run it, and query it.
# Uses port 9697 by default so it can run next to the local Q4 server.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-zoomcamp-model:2026-hw5}"
PORT="${PORT:-9697}"
NAME="hw5-q6-$$"

docker build -t "$IMAGE" .
docker image inspect "$IMAGE" --format 'image size: {{.Size}} bytes (not graded)'

docker run -d --rm --name "$NAME" -p "$PORT:9696" "$IMAGE" >/dev/null
trap 'docker stop "$NAME" >/dev/null' EXIT

python3 solution/q4_client.py "http://localhost:$PORT" --save solution/results/q6_container_response.json
# The course's own smoke client, pointed at the container.
python3 q6_test.py "http://localhost:$PORT/predict"
