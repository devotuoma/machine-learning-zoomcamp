# My ML API Said "200 OK" to TikTok

*ML Zoomcamp week 5: deploying a model with pickle, FastAPI, uv and Docker, then finding out what it accepts. A 4-minute read.*

**Code:** https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw05-deployment/cohorts/2026/homework/05-deployment/solution

---

Training a model is half the job. In week 5 of [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp), the model is already trained and frozen (`pipeline.bin`: a `DictVectorizer` plus a `LogisticRegression` that predicts whether a lead will sign up). The task is to ship it: load it, serve it and containerize it, getting the same answer at every step.

## Step 0: trust, then unpickle

`pickle.load` can run arbitrary code, so the first thing I wrote was a checksum check:

```python
assert sha256("pipeline.bin") == metadata["model_sha256"]   # 1646bbdc…
```

Then the environment. `uv sync --locked` installs exactly what `uv.lock` pins: Python 3.11.15 and **scikit-learn 1.7.2** (Q2). Running `uv sync` without `--locked` can quietly upgrade packages, and then your probabilities drift.

## Three ways to ask the same question

**Q3: the pickled model directly.**

```python
pipeline = pickle.load(open("pipeline.bin", "rb"))
pipeline.predict_proba([normalize_record(lead)])[0, 1]   # 0.533
```

`normalize_record` matters. It's the same preprocessing the API uses (training medians for missing numbers, `"NA"` for missing categories). If you write your own `fillna(0)`, you get a different answer.

**Q4: FastAPI.** `uv run uvicorn predict:app --port 9696`, then POST a second lead → **0.770**. `/health` returns the model's checksum, so you can confirm which artifact is being served.

**Q5–Q6: Docker.** The Dockerfile pins `python:3.11.15-slim-bookworm` by digest, copies `uv` 0.10.11 and runs `uv sync --locked`. Build, run, POST → **0.770** again.

Same artifact + same lockfile + same Python + same preprocessing = same number. My local `uv` (0.13.6) differed from the image's (0.10.11), and it made no difference.

## Why 0.533?

A logistic regression is a sum of `weight × value` terms passed through a sigmoid, so you can open any prediction up:

![Contributions](../images/q3_contributions.png)

`interaction_count` adds +0.99 and `annual_income` subtracts −0.98. `lead_score`, the strongest single feature in week 4, adds only +0.009. The pipeline has no scaler, so raw income dominates. Keep this lead fixed and raise only the income, and the prediction flips to "won't convert" at about 90k, even though converters earn *more* on average:

![Income](../images/income_sensitivity.png)

## Then I sent it realistic input

| request | status | P |
|---|---|---|
| the Q3 lead | 200 | 0.533 |
| `lead_source: "Paid_Ads"` | 200 | 0.553 |
| `lead_source: "tiktok"` | 200 | 0.553 |
| `annual_income: 7927600` (typo ×100) | 200 | **0.000** |
| `interaction_count: -3` | 200 | 0.167 |
| `number_of_courses_viewed: 2.5` | 422 | — |

Two things combine here:

1. **`DictVectorizer` ignores categories it didn't see in training.** `"Paid_Ads"` and `"tiktok"` match no column, so the `lead_source` block becomes all zeros. Both get the same score, and no error is raised.
2. **Pydantic validates types, not values.** A float is a float, even a negative one or eight million. Only `2.5` fails, because it isn't an integer.

## The fix: validate values, using the model's own vocabulary

I didn't touch the frozen pipeline. I added a stricter request model next to it:

```python
def vocabulary(feature):
    names = pipeline.steps[0][1].get_feature_names_out()
    return tuple(n.split("=", 1)[1] for n in names
                 if n.startswith(f"{feature}=") and not n.endswith("=NA"))

class Numbers(BaseModel):
    model_config = ConfigDict(extra="forbid")
    interaction_count: Optional[int] = Field(None, ge=0)
    annual_income: Optional[float] = Field(None, ge=0, lt=1_000_000)
    lead_score: Optional[float] = Field(None, ge=0, le=1)
    ...

StrictLead = create_model("StrictLead", __base__=Numbers,
    **{f: (Optional[Literal[vocabulary(f)]], None) for f in CATEGORICAL_FEATURES})
```

A `before` validator lowercases and replaces spaces first, so honest typos are fixed rather than rejected:

- `"Paid_Ads"`, `"North America"` → **0.533**, the same as the clean lead
- `"tiktok"`, `-3`, the income typo, `lead_score: 4.1`, `leadscore` (misspelled field) → **422** with a clear message
- income of 250,000 → a prediction plus a warning that it's outside the training range

Because the allowed values come from the `DictVectorizer`, the API can't fall out of sync with the model. 23 pytest checks start real uvicorn servers and test all of this.

## Takeaways

1. **Check the artifact before unpickling.** A checksum takes one line.
2. **Use `uv sync --locked`.** The lockfile is what makes results reproducible.
3. **Share one preprocessing function** between offline scripts and the API.
4. **Validate values, not just types.** 200 OK doesn't mean the input made sense.
5. **Open up your model.** A contribution chart showed that income, not `lead_score`, drives the predictions.

**Answers:** uv 0.13.6 · scikit-learn 1.7.2 · 0.533 · 0.770 · python:3.11.15-slim-bookworm · 0.770

**Full code and tests:** https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw05-deployment/cohorts/2026/homework/05-deployment/solution

*Thanks to Alexey Grigorev and DataTalks.Club for ML Zoomcamp.*

*Tags: Machine Learning, MLOps, FastAPI, Docker, Python*
