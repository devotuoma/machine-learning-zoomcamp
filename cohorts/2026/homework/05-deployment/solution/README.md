# Homework 5 — Deploying Machine Learning Models (2026)

Solution for [Homework 5](../homework.md) of ML Zoomcamp 2026. We take the frozen lead-scoring model (`pipeline.bin`: `DictVectorizer` + `LogisticRegression`), check it, load it, serve it with FastAPI, and run it in the course's Docker image. Then we test what the service does with awkward input.

## Files

| File | What it does |
|---|---|
| [q1_env_check.sh](q1_env_check.sh) | Q1: local `uv` version, pinned Python, Dockerfile `uv`, artifact hash |
| [q2_locked_version.py](q2_locked_version.py) | Q2: reads scikit-learn from `uv.lock` and `pyproject.toml` |
| [verify_artifact.py](verify_artifact.py) | Checks `pipeline.bin` against its SHA-256 before anything unpickles it |
| [q3_load_model.py](q3_load_model.py) | Q3: `pickle.load` + `normalize_record` + `predict_proba` |
| [q4_serve.sh](q4_serve.sh) / [q4_client.py](q4_client.py) | Q4: start uvicorn, call `/health` and `/predict` |
| [q5_dockerfile_base.py](q5_dockerfile_base.py) | Q5: parses the `FROM` line |
| [q6_container.sh](q6_container.sh) | Q6: build, run and query the container |
| [explain_prediction.py](explain_prediction.py) | Splits a prediction into per-feature log-odds contributions |
| [income_sensitivity.py](income_sensitivity.py) | What-if: same lead, income from 0 to 200k |
| [edge_cases.py](edge_cases.py) | Sends typos, unseen categories and bad numbers to the API |
| [solve_all.py](solve_all.py) | Writes every answer to [results/answers.json](results/answers.json) |
| [tests/](tests/) | 13 pytest checks, including a live uvicorn server |
| [Makefile](Makefile) | `make q3`, `make serve`, `make q4`, `make q6`, `make test`, … |
| [deliverables/](deliverables/) | Learning-in-public posts and FAQ proposal |

## Run it

```bash
cd cohorts/2026/homework/05-deployment
uv sync --locked                      # Python 3.11.15, scikit-learn 1.7.2
python3 solution/verify_artifact.py   # always before unpickling
uv run --locked python solution/q3_load_model.py

./solution/q4_serve.sh                # terminal 1
python3 solution/q4_client.py         # terminal 2

./solution/q6_container.sh            # Docker
uv run --locked --with pytest pytest solution/tests -q
```
