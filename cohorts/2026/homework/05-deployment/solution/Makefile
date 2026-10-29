# Homework 5 shortcuts. Run from this directory: `make all`.
HW := ..
UV_RUN := cd $(HW) && uv run --locked
URL ?= http://localhost:9696

.PHONY: all sync q1 q2 verify q3 serve q4 q5 q6 explain edges plots test

all: sync q1 q2 verify q3 q5 test

sync:
	cd $(HW) && uv sync --locked

q1:
	./q1_env_check.sh

q2:
	python3 q2_locked_version.py

verify:
	python3 verify_artifact.py

q3:
	$(UV_RUN) python solution/q3_load_model.py

serve:            ## blocks; run in its own terminal
	./q4_serve.sh

q4:               ## needs `make serve` running
	python3 q4_client.py $(URL) --save results/q4_api_response.json

q5:
	python3 q5_dockerfile_base.py

q6:               ## needs Docker
	./q6_container.sh

explain:
	$(UV_RUN) python solution/explain_prediction.py solution/leads/q3_lead.json
	$(UV_RUN) python solution/explain_prediction.py solution/leads/q4_lead.json

edges:            ## needs `make serve` running
	python3 edge_cases.py $(URL) --save results/edge_cases.json

plots:
	$(UV_RUN) --with matplotlib python solution/explain_prediction.py solution/leads/q3_lead.json --plot solution/images/q3_contributions.png
	$(UV_RUN) --with matplotlib python solution/explain_prediction.py solution/leads/q4_lead.json --plot solution/images/q4_contributions.png
	$(UV_RUN) --with matplotlib python solution/income_sensitivity.py --plot solution/images/income_sensitivity.png

test:
	$(UV_RUN) --with pytest pytest solution/tests -q
