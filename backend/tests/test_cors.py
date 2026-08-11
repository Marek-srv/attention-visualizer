from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import DEFAULT_CORS_ORIGINS, app, configure_cors, get_cors_origins


def test_default_local_origins_receive_cors_headers() -> None:
    client = TestClient(app)

    for origin in DEFAULT_CORS_ORIGINS:
        response = client.get("/api/health", headers={"Origin": origin})

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin
        assert response.headers["access-control-allow-credentials"] == "true"


def test_configured_deployed_origin_receives_cors_headers(monkeypatch) -> None:
    deployed_origin = "https://attention.example.com"
    monkeypatch.setenv(
        "CORS_ORIGINS",
        f" , {deployed_origin},, {deployed_origin}, http://localhost:5173, ",
    )
    configured_origins = get_cors_origins()
    configured_app = FastAPI()
    configure_cors(configured_app)

    @configured_app.get("/probe")
    def probe() -> dict[str, bool]:
        return {"ok": True}

    response = TestClient(configured_app).get("/probe", headers={"Origin": deployed_origin})

    assert configured_origins.count(deployed_origin) == 1
    assert configured_origins.count("http://localhost:5173") == 1
    assert "" not in configured_origins
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == deployed_origin
    assert response.headers["access-control-allow-credentials"] == "true"


def test_unlisted_origin_does_not_receive_allow_origin_header(monkeypatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "https://attention.example.com")
    configured_app = FastAPI()
    configure_cors(configured_app)

    @configured_app.get("/probe")
    def probe() -> dict[str, bool]:
        return {"ok": True}

    response = TestClient(configured_app).get(
        "/probe", headers={"Origin": "https://unlisted.example.com"}
    )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
