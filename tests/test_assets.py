from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_assets():
    resp = client.get("/assets")
    assert resp.status_code == 200
    data = resp.json()
    assert "tickers" in data
    assert isinstance(data["tickers"], list)
    assert len(data["tickers"]) > 0
