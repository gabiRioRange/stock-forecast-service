import argparse
from typing import List

from ml.data_prep import prepare_dataset
from ml.models import train_lstm_regressor, train_sklearn_regressor


def train_for_ticker(
    ticker: str,
    model_type: str = "random_forest",
    start: str = "2018-01-01",
    end: str = "2025-01-01",
):
    """Baixa dados, prepara features e treina modelo para um ticker."""
    print(f"[{ticker}] Preparando dados...")
    X, y = prepare_dataset(ticker, start=start, end=end)
    print(f"[{ticker}] Treinando modelo {model_type}...")
    if model_type in ["linear", "random_forest"]:
        result = train_sklearn_regressor(X, y, ticker=ticker, model_type=model_type)
    elif model_type == "lstm":
        result = train_lstm_regressor(X, y, ticker=ticker)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    metrics = result["metrics"]
    print(
        f"[{ticker}] {model_type} -> MAE={metrics.mae:.2f}, RMSE={metrics.rmse:.2f}"
    )
    print(f"[{ticker}] Modelo salvo em: {result['model_path']}\n")


def main(
    tickers: List[str],
    model_type: str = "random_forest",
    start: str = "2018-01-01",
    end: str = "2025-01-01",
):
    """Treina modelos para uma lista de tickers."""
    for ticker in tickers:
        train_for_ticker(ticker, model_type=model_type, start=start, end=end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina modelos de previsão de ações")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "PETR4.SA"],
        help="Lista de tickers para treinar",
    )
    parser.add_argument(
        "--model-type",
        choices=["linear", "random_forest", "lstm"],
        default="random_forest",
        help="Tipo de modelo",
    )
    parser.add_argument("--start", default="2018-01-01", help="Data inicial")
    parser.add_argument("--end", default="2025-01-01", help="Data final")

    args = parser.parse_args()
    main(args.tickers, args.model_type, args.start, args.end)
