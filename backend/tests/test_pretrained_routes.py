from fastapi.testclient import TestClient

from app.main import app
from app.pretrained_service import pretrained_service


client = TestClient(app)


def test_pretrained_status_does_not_load_a_model() -> None:
    response = client.get("/api/pretrained/status")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == pretrained_service.model_name
    assert isinstance(body["loaded"], bool)
    assert "dependencies_available" in body


def test_pretrained_requests_validate_without_downloading() -> None:
    assert client.post("/api/pretrained/predict", json={"text": "   ", "top_k": 5}).status_code == 422
    assert client.post(
        "/api/pretrained/inspect",
        json={"text": "I love", "layer": -1, "head": 0, "query_token": 0, "top_k": 5},
    ).status_code == 422
    assert client.post(
        "/api/pretrained/load", json={"model_name": "not-the-configured-model"}
    ).status_code == 422
