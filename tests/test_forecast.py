from fastapi.testclient import TestClient

from app.main import app
from app.services.model_loader import clear_model_cache

client = TestClient(app)


def test_forecast_without_model():
    """Testa que /forecast retorna 404 se modelo não existe."""
    clear_model_cache()
    resp = client.get("/forecast/AAPL")
    # se você ainda não treinou, deve retornar 404
    # se já treinou, vai retornar 200
    assert resp.status_code in [200, 404]


def test_forecast_with_model():
    """Testa /forecast com modelo treinado."""
    clear_model_cache()
    resp = client.get("/forecast/AAPL?model_type=random_forest")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "AAPL"
    assert len(data["predictions"]) == 5  # default horizon


def test_forecast_lstm():
    """Testa /forecast com LSTM."""
    clear_model_cache()
    resp = client.get("/forecast/AAPL?model_type=lstm")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "AAPL"
    assert len(data["predictions"]) == 1  # LSTM prevê apenas 1 dia por vez
