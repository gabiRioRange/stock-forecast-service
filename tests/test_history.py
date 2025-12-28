from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_history():
    """Testa /history para um ticker."""
    resp = client.get("/history/AAPL")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "AAPL"
    assert len(data["items"]) > 0
    assert "date" in data["items"][0]
    assert "close" in data["items"][0]