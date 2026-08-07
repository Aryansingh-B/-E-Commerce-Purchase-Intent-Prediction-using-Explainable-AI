"""API tests (PRD §4.7, §5) — require trained artifacts to exist.

Run `python -m src.train && python -m src.train_explainers` first (CI does
this before pytest — see .github/workflows/ci.yml).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.config.settings import get_settings

VALID_PAYLOAD = {
    "Administrative": 2,
    "Administrative_Duration": 40.5,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 25,
    "ProductRelated_Duration": 620.3,
    "BounceRates": 0.01,
    "ExitRates": 0.03,
    "PageValues": 12.4,
    "SpecialDay": 0.0,
    "Month": "Nov",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": False,
}


@pytest.fixture(scope="module")
def artifacts_exist() -> bool:
    settings = get_settings()
    return settings.paths.resolve("model_artifact").exists()


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health(client, artifacts_exist):
    if not artifacts_exist:
        pytest.skip("Model artifact not trained yet — run `python -m src.train` first.")
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valid_payload(client, artifacts_exist):
    if not artifacts_exist:
        pytest.skip("Model artifact not trained yet — run `python -m src.train` first.")
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["conversion_probability"] <= 1.0
    assert body["confidence"] in ("low", "medium", "high")


def test_predict_rejects_invalid_visitor_type(client, artifacts_exist):
    if not artifacts_exist:
        pytest.skip("Model artifact not trained yet — run `python -m src.train` first.")
    bad = dict(VALID_PAYLOAD, VisitorType="Alien")
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_out_of_range_rate(client, artifacts_exist):
    if not artifacts_exist:
        pytest.skip("Model artifact not trained yet — run `python -m src.train` first.")
    bad = dict(VALID_PAYLOAD, BounceRates=1.5)
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_handles_unknown_traffic_type_category(client, artifacts_exist):
    """Unseen category at serving time must degrade gracefully, not crash (PRD §4.7)."""
    if not artifacts_exist:
        pytest.skip("Model artifact not trained yet — run `python -m src.train` first.")
    unseen = dict(VALID_PAYLOAD, TrafficType=9999)
    r = client.post("/predict", json=unseen)
    assert r.status_code == 200
