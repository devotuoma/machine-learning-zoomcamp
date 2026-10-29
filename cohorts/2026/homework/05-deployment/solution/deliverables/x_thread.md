# X — Homework 5

## Thread (4 posts, each under 280 characters)

**1/**
I sent "tiktok" to my ML API. It replied 200 OK. 🧵

#mlzoomcamp week 5: pickle → FastAPI → Docker, with a locked uv environment.

Script, API and container all agree: 0.770 ✅

**2/**
Then I sent realistic mistakes:

"Paid_Ads" → 200
"tiktok" → 200 (same score as the typo)
income ×100 → 200, P = 0.000
interaction_count -3 → 200

DictVectorizer drops unknown categories. Pydantic checks types, not values.

**3/**
Fix: Literal types built from the model's own DictVectorizer vocabulary + Field(ge=0) bounds.

"Paid_Ads" → normalised, 0.533
"tiktok" → 422
income typo → 422

23 tests, frozen model untouched.

**4/**
Lesson: deployment isn't done when the container starts. Validate the input.

Code: https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw05-deployment/cohorts/2026/homework/05-deployment/solution

Thanks @Al_Grigor @DataTalksClub 🙏

## Single post version

My ML API said 200 OK to "tiktok" and returned P = 0.000 for an income typo. 🙃

Fix: Literal + Field bounds from the model's own vocabulary → 422.

https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw05-deployment/cohorts/2026/homework/05-deployment/solution

#mlzoomcamp #FastAPI

## X Spaces talking points (5 minutes)

**Title:** "My API said yes to tiktok"

1. **Hook:** live-demo the "tiktok" request → 200 OK.
2. **Reproducibility:** checksum → `uv sync --locked` → same 0.770 locally, in the API and in Docker.
3. **Why bad input passes:** DictVectorizer ignores unseen categories; Pydantic checks types only.
4. **Fix:** Literal from the model vocabulary, Field bounds, `extra="forbid"`.
5. **Close:** "Validate the input, not just the container." Share the repo link.
