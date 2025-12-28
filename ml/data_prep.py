from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_history(
    ticker: str,
    start: str = "2018-01-01",
    end: str = "2025-01-01",
) -> pd.DataFrame:
    """Baixa histórico do ticker via yfinance e salva em data/raw."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.reset_index()
    df.columns = df.columns.droplevel(1)  # Remove ticker level
    df.rename(columns={"Date": "date"}, inplace=True)
    df.to_csv(RAW_DIR / f"{ticker}_raw.csv", index=False)
    return df


def clean_and_engineer_features(
    df: pd.DataFrame,
    window_short: int = 5,
    window_long: int = 20,
) -> pd.DataFrame:
    """Cria features: retornos, médias móveis, volatilidade, RSI, MACD."""
    df = df.copy()
    df.sort_values("date", inplace=True)
    df["close"] = df["Close"]
    df["return_1d"] = df["close"].pct_change()
    df["ma_short"] = df["close"].rolling(window_short).mean()
    df["ma_long"] = df["close"].rolling(window_long).mean()
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


def make_supervised(
    df: pd.DataFrame,
    target_horizon: int = 1,
    feature_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Monta dataset supervisionado (X, y) para prever close em t+horizon."""
    if feature_cols is None:
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
    df = df.copy()
    df[f"target_close_t+{target_horizon}"] = df["close"].shift(-target_horizon)
    df = df.dropna().reset_index(drop=True)

    X = df[feature_cols]
    y = df[f"target_close_t+{target_horizon}"]
    return X, y


def prepare_dataset(
    ticker: str,
    start: str = "2018-01-01",
    end: str = "2025-01-01",
    target_horizon: int = 1,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Pipeline completo: baixa, limpa, cria features e dataset supervisionado."""
    raw_path = RAW_DIR / f"{ticker}_raw.csv"
    if raw_path.exists():
        df = pd.read_csv(raw_path, parse_dates=["date"])
        # Ensure numeric columns are float
        numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        df = download_history(ticker, start=start, end=end)

    df = clean_and_engineer_features(df)
    X, y = make_supervised(df, target_horizon=target_horizon)

    processed_path = PROCESSED_DIR / f"{ticker}_processed.csv"
    df_proc = X.copy()
    df_proc["target"] = y
    df_proc.to_csv(processed_path, index=False)
    return X, y
