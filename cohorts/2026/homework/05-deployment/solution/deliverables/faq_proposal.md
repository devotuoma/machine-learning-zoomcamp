Proposed issue for https://github.com/DataTalksClub/faq/issues/new?template=faq-proposal.yml

**Title:** [FAQ] Homework 5 (2026): Why does `/predict` return 200 OK for `"Paid_Ads"`, `"tiktok"` or a negative count, and how do I reject bad input?

### Course

machine-learning-zoomcamp

### Question

Homework 5 (2026): I sent `"lead_source": "Paid_Ads"` (capital letters) to the API and got HTTP 200 with a different probability than for `"paid_ads"`. `"tiktok"`, `"interaction_count": -3` and an income typo of `7927600` also return 200. Shouldn't the API reject these?

### Answer

The course API validates **types**, not **values**, and `DictVectorizer` silently ignores categories it didn't see in training. Your graded answers are unaffected: the homework leads are all valid. In a real service, though, these mistakes go unnoticed.

What happens with the Q3 lead (`/predict` from the provided `predict.py`):

| change | status | probability |
|---|---|---|
| none | 200 | 0.533 |
| `lead_source: "Paid_Ads"` | 200 | 0.553 |
| `lead_source: "tiktok"` | 200 | 0.553 |
| `annual_income: 7927600` (×100 typo) | 200 | **0.000** |
| `interaction_count: -3` | 200 | 0.167 |
| `number_of_courses_viewed: 2.5` | 422 | — |

**Why:**
- `DictVectorizer` one-hot encodes only the `feature=value` pairs it saw during `fit`. `"Paid_Ads"` and `"tiktok"` match no column, so the whole `lead_source` block becomes zeros, and both get the same score.
- Pydantic checks that `annual_income` is a float and that `interaction_count` is an int. Without `Field` constraints, any float or int passes. Only `2.5` fails, because it isn't a valid int.
- Income isn't scaled, so a ×100 typo multiplies a −1.2e-5 weight by about 8 million and pushes the probability to 0.

**Fix: validate values in the request model.** Read the allowed categories from the model itself so they can't drift:

```python
from typing import Literal, Optional
from pydantic import BaseModel, Field, create_model, field_validator

def vocabulary(feature):
    names = pipeline.steps[0][1].get_feature_names_out()
    return tuple(n.split("=", 1)[1] for n in names
                 if n.startswith(f"{feature}=") and not n.endswith("=NA"))

class Numbers(BaseModel):
    number_of_courses_viewed: Optional[int] = Field(None, ge=0)
    annual_income: Optional[float] = Field(None, ge=0, lt=1_000_000)
    interaction_count: Optional[int] = Field(None, ge=0)
    lead_score: Optional[float] = Field(None, ge=0, le=1)

    @field_validator("lead_source", "industry", "employment_status", "location",
                     mode="before", check_fields=False)
    @classmethod
    def clean(cls, v):  # "Paid_Ads" -> "paid_ads", "North America" -> "north_america"
        if isinstance(v, str):
            return v.strip().lower().replace(" ", "_") or None
        return v

Lead = create_model("Lead", __base__=Numbers, **{
    f: (Optional[Literal[vocabulary(f)]], None)
    for f in ["lead_source", "industry", "employment_status", "location"]})
```

Now `"Paid_Ads"` scores the same as `"paid_ads"` (0.533). `"tiktok"`, negative counts, the income typo and `lead_score: 4.1` return **422** with a clear message, and missing fields still fall back to the training medians. Add `model_config = ConfigDict(extra="forbid")` to reject misspelled field names such as `leadscore` too.

Don't change the provided `predict.py` for the graded Q4/Q6 runs. Use a separate app for experiments.

Full working version with 23 tests: https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw05-deployment/cohorts/2026/homework/05-deployment/solution

### Checklist

- [x] I have searched existing FAQs and this question is not already answered
- [x] The answer provides accurate, helpful information
- [x] I have included any relevant code examples or links
