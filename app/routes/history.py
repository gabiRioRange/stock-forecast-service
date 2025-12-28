from datetime import date
from typing import Optional

import yfinance as yf
from fastapi import APIRouter, HTTPException, Query

from app.config import SUPPORTED_TICKERS
from app.schemas import HistoryItem, HistoryResponse

router = APIRouter()


@router.get("/history/{ticker}", response_model=HistoryResponse)
def get_history(
    ticker: str,
    start: Optional[date] = Query(None, description="Data inicial (YYYY-MM-DD)"),
    end: Optional[date] = Query(None, description="Data final (YYYY-MM-DD)"),
    limit: int = Query(60, ge=1, le=252, description="Número de dias"),
):
    """Retorna histórico recente de um ticker."""
    if ticker not in SUPPORTED_TICKERS:
        raise HTTPException(status_code=404, detail="Ticker not supported")

    df = yf.download(
        ticker,
        start=start.isoformat() if start else None,
        end=end.isoformat() if end else None,
        progress=False,
    )

    if df.empty:
        raise HTTPException(status_code=404, detail="No data for ticker")

    df.columns = df.columns.droplevel(1)  # Remove ticker level
    df = df.tail(limit).reset_index()
    items = [
        HistoryItem(
            date=row["Date"].date(),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=int(row["Volume"]),
        )
        for _, row in df.iterrows()
    ]

    return HistoryResponse(ticker=ticker, items=items)
