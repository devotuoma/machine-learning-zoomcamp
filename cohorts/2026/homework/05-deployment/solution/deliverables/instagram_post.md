# Instagram carousel — Homework 5

**Format:** 5 slides, 1080×1350. Plots from [../images/](../images/).

| Slide | Visual | Text |
|---|---|---|
| 1 | Terminal: `"lead_source": "tiktok"` → `200 OK` | **My ML API said yes to TikTok** |
| 2 | 📦 pickle → ⚡ FastAPI → 🐳 Docker | Same answer everywhere: **0.770** |
| 3 | Table: Paid_Ads / tiktok / income ×100 → all 200 | **Typos pass. An income typo → P = 0.000** |
| 4 | `income_sensitivity.png` | Same lead, higher income → flips to "no" at ~90k |
| 5 | Code: `Literal[...]` + `Field(ge=0)` → `422` | **Validate the input, not just the container** ✅ |

## Caption

My ML API said 200 OK to "tiktok". 🙃

Week 5 of #mlzoomcamp: I deployed a lead-scoring model with pickle, FastAPI, uv and Docker. All three gave the same answer (0.770).

Then I sent typos. "Paid_Ads", "tiktok", an income ×100 → all accepted, one returned a 0% probability. The fix: validate values, not just types. Now nonsense gets a 422.

Code: https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw05-deployment/cohorts/2026/homework/05-deployment/solution (also in bio)

Thanks @datatalksclub 🙏

#machinelearning #mlops #python #docker #fastapi #learninpublic #mlzoomcamp
