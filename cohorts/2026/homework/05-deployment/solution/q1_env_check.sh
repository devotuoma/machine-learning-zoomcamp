#!/usr/bin/env bash
# Question 1: record the local toolchain. Not graded: the uv version is a
# property of your machine, not of the model.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "uv:              $(uv --version)"
echo "pinned python:   $(cat .python-version)"
echo "dockerfile uv:   $(grep -o 'astral-sh/uv:[0-9.]*' Dockerfile | cut -d: -f2)"
echo "pipeline.bin:    $(sha256sum pipeline.bin | cut -d' ' -f1)"
