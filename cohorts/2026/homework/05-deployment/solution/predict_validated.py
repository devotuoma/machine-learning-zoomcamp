"""A stricter /predict for the same frozen pipeline (not part of the graded run).

The course API validates types only. This version also validates values:
- categories must be in the model's own DictVectorizer vocabulary, after
  harmless normalisation ("Paid_Ads" -> "paid_ads", "North America" -> "north_america");
  unknown values such as "tiktok" get 422 instead of a silent all-zero one-hot block
- counts and income must be >= 0, lead_score in [0, 1], income below 1,000,000
- values outside the training range still get a prediction, plus a warning

Run (from 05-deployment):
    uv run --locked uvicorn --app-dir solution predict_validated:app --port 9698
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import CATEGORICAL_FEATURES, normalize_record  # noqa: E402
from predict import pipeline  # noqa: E402  (same checked artifact as the course API)

# Training-data ranges from course_lead_scoring_2026.csv
TRAINING_RANGE = {
    "number_of_courses_viewed": (0, 6),
    "annual_income": (12_000, 109_886),
    "interaction_count": (0, 13),
    "lead_score": (0.01, 0.99),
}


def vocabulary(feature: str) -> tuple[str, ...]:
    """Category values the model actually learned, read from the DictVectorizer."""
    prefix = f"{feature}="
    names = pipeline.steps[0][1].get_feature_names_out()
    return tuple(n.removeprefix(prefix) for n in names if n.startswith(prefix) and n != f"{prefix}NA")


def _clean(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip().lower().replace(" ", "_").replace("-", "_")
        return value or None
    return value


class _Numbers(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number_of_courses_viewed: Optional[int] = Field(None, ge=0)
    annual_income: Optional[float] = Field(None, ge=0, lt=1_000_000)
    interaction_count: Optional[int] = Field(None, ge=0)
    lead_score: Optional[float] = Field(None, ge=0, le=1)

    @field_validator(*CATEGORICAL_FEATURES, mode="before", check_fields=False)
    @classmethod
    def normalise_category(cls, v: Any) -> Any:
        return _clean(v)


StrictLead = create_model(
    "StrictLead",
    __base__=_Numbers,
    **{f: (Optional[Literal[vocabulary(f)]], None) for f in CATEGORICAL_FEATURES},
)

app = FastAPI(title="ML Zoomcamp 2026 lead scoring (validated)")


@app.post("/predict")
def predict(lead: StrictLead) -> dict[str, Any]:  # type: ignore[valid-type]
    raw = lead.model_dump()
    warnings = [
        f"{name}={raw[name]} is outside the training range {lo}-{hi}"
        for name, (lo, hi) in TRAINING_RANGE.items()
        if raw[name] is not None and not lo <= raw[name] <= hi
    ]
    probability = float(pipeline.predict_proba([normalize_record(raw)])[0, 1])
    return {
        "conversion_probability": round(probability, 6),
        "conversion": probability >= 0.5,
        "warnings": warnings,
    }
