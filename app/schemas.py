from datetime import date
from typing import List

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class AssetListResponse(BaseModel):
    tickers: List[str]


class HistoryItem(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


class HistoryResponse(BaseModel):
    ticker: str
    items: List[HistoryItem]


class ForecastItem(BaseModel):
    date: date
    close: float


class ForecastResponse(BaseModel):
    ticker: str
    horizon_days: int
    predictions: List[ForecastItem]
