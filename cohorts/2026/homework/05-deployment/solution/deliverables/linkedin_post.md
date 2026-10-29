# LinkedIn post — Homework 5

I typed "tiktok" into my ML API. It said 200 OK. 🙃

#mlzoomcamp week 5: deploying a lead-scoring model with pickle, FastAPI, uv and Docker.

The homework part was smooth:
✅ Locked env (scikit-learn 1.7.2, uv sync --locked)
✅ Checked the model's SHA-256 before unpickling
✅ Same prediction from the script, the API and the container: 0.770

Then I tried input a real user might send:
🔸 "Paid_Ads" instead of "paid_ads" → 200, different score
🔸 "tiktok" (never seen) → 200, same score as the typo
🔸 Income typo ×100 → 200, probability 0.000
🔸 interaction_count = -3 → 200

DictVectorizer silently drops unknown categories, and Pydantic checks types, not values.

The fix: Literal types built from the model's own vocabulary, plus Field bounds. Typos get normalised, nonsense gets a 422. 23 tests.

Lesson: a model in production is only as good as the input it accepts.

Code 👉 https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw05-deployment/cohorts/2026/homework/05-deployment/solution

Thanks @Alexey Grigorev and @DataTalks.Club 🙏

#MachineLearning #MLOps #FastAPI #Docker #Python #LearningInPublic #mlzoomcamp
