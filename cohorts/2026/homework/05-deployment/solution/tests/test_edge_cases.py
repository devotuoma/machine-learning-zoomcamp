"""Document current API behaviour on awkward input (not graded)."""

from edge_cases import BASE, post


def test_unseen_category_is_silently_accepted(api_url):
    status, unseen = post(f"{api_url}/predict", {**BASE, "lead_source": "tiktok"})
    _, typo = post(f"{api_url}/predict", {**BASE, "lead_source": "Paid_Ads"})
    _, real = post(f"{api_url}/predict", BASE)
    assert status == 200
    assert unseen == typo  # both become an all-zero lead_source block
    assert unseen["conversion_probability"] != real["conversion_probability"]


def test_fractional_course_count_is_rejected(api_url):
    status, _ = post(f"{api_url}/predict", {**BASE, "number_of_courses_viewed": 2.5})
    assert status == 422


def test_income_typo_collapses_probability(api_url):
    _, body = post(f"{api_url}/predict", {**BASE, "annual_income": BASE["annual_income"] * 100})
    assert body["conversion_probability"] < 0.001
