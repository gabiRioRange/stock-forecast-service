from datetime import timedelta
from pathlib import Path
from typing import List

import joblib
import pandas as pd
import torch
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query

from app.config import DEFAULT_FORECAST_HORIZON_DAYS, SUPPORTED_TICKERS
from app.schemas import ForecastItem, ForecastResponse
from app.services.model_loader import get_model

router = APIRouter()


def build_features_from_recent_history(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features a partir de histórico recente (mesmo pipeline do treino)."""
    df = df.reset_index()
    df.rename(columns={"Date": "date"}, inplace=True)
    df["close"] = df["Close"]
    df["return_1d"] = df["close"].pct_change()
    df["ma_short"] = df["close"].rolling(5).mean()
    df["ma_long"] = df["close"].rolling(20).mean()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    df = df.dropna().reset_index(drop=True)
    return df


@router.get("/forecast/{ticker}", response_model=ForecastResponse)
def forecast(
    ticker: str,
    horizon_days: int = Query(
        DEFAULT_FORECAST_HORIZON_DAYS,
        ge=1,
        le=30,
        description="Número de dias para prever",
    ),
    model_type: str = Query("random_forest", description="Tipo de modelo: random_forest ou lstm"),
):
    """Gera previsão de fechamento para os próximos N dias."""
    if ticker not in SUPPORTED_TICKERS:
        raise HTTPException(status_code=404, detail="Ticker not supported")

    # baixa histórico recente (6 meses suficiente para features)
    df = yf.download(ticker, period="6mo", progress=False)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data for ticker")

    df_feat = build_features_from_recent_history(df)
    if df_feat.empty:
        raise HTTPException(status_code=500, detail="Not enough data for features")

    # última linha de features
    last_row = df_feat.iloc[-1]
    feature_cols = [
        "close",
        "return_1d",
        "ma_short",
        "ma_long",
        "volatility_10d",
        "rsi",
        "macd",
        "macd_signal",
    ]

    try:
        model = get_model(ticker, model_type=model_type)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found for {ticker} with {model_type}. Run ml/train.py first.",
        )

    predictions: List[ForecastItem] = []
    last_date = df_feat["date"].iloc[-1].date()
    if model_type == "lstm":
        # Para LSTM, usar sequências
        seq_length = 20
        if len(df_feat) < seq_length:
            raise HTTPException(status_code=500, detail="Not enough data for LSTM prediction")
        last_seq = df_feat[feature_cols].iloc[-seq_length:].values
        # Normalizar features
        scaler_X_path = Path("ml/models") / f"{ticker}_lstm_scaler_X.pkl"
        scaler_y_path = Path("ml/models") / f"{ticker}_lstm_scaler_y.pkl"
        if scaler_X_path.exists() and scaler_y_path.exists():
            scaler_X = joblib.load(scaler_X_path)
            scaler_y = joblib.load(scaler_y_path)
            last_seq = scaler_X.transform(last_seq)
        last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            pred_scaled = model(last_seq).item()
        # Desnormalizar prediction
        if scaler_y_path.exists():
            pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        else:
            pred = pred_scaled
        predictions.append(ForecastItem(date=last_date + timedelta(days=1), close=pred))
    else:
        # Para sklearn
        current_features = last_row[feature_cols].values.reshape(1, -1)
        # loop simples: prevê próximos N dias usando a última janela
        # (em produção você pode fazer um loop autoregressivo mais sofisticado)
        for step in range(1, horizon_days + 1):
            pred_close = float(model.predict(current_features)[0])
            forecast_date = last_date + timedelta(days=step)
            predictions.append(ForecastItem(date=forecast_date, close=pred_close))

    return ForecastResponse(
        ticker=ticker,
        horizon_days=horizon_days,
        predictions=predictions,
    )
