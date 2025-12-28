import sys
from pathlib import Path

# Adicionar o diretório raiz ao sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from fastapi import FastAPI

from app.routes import assets, forecast, health, history

app = FastAPI(
    title="Stock Forecast Service",
    version="0.1.0",
    description="API de previsão de preços de ações com yfinance + scikit-learn",
)

app.include_router(health.router, tags=["Health"])
app.include_router(assets.router, tags=["Assets"])
app.include_router(history.router, tags=["History"])
app.include_router(forecast.router, tags=["Forecast"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
