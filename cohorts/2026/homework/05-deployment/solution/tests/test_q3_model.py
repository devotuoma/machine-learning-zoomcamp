import math

import pytest

from explain_prediction import contributions
from q3_load_model import conversion_probability, load_pipeline


@pytest.fixture(scope="module")
def pipeline():
    return load_pipeline()


def test_q3_probability(pipeline, lead_q3):
    assert conversion_probability(pipeline, lead_q3) == pytest.approx(0.533, abs=0.0005)


def test_inference_is_deterministic(pipeline, lead_q3):
    assert conversion_probability(pipeline, lead_q3) == conversion_probability(pipeline, lead_q3)


def test_contributions_reproduce_predict_proba(pipeline, lead_q3):
    intercept, parts = contributions(pipeline, lead_q3)
    p = 1 / (1 + math.exp(-(intercept + sum(parts.values()))))
    assert p == pytest.approx(conversion_probability(pipeline, lead_q3), abs=1e-12)
