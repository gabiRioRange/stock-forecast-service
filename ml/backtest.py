import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ml.data_prep import prepare_dataset
from ml.models import load_model


def walk_forward_validation(
    ticker: str,
    model_type: str = "random_forest",
    train_window: int = 252,  # ~1 ano
    test_window: int = 21,    # ~1 mês
    step: int = 21,           # avançar 1 mês
) -> Dict[str, List[float]]:
    """
    Realiza walk-forward validation para avaliar o modelo em diferentes períodos.
    """
    print(f"[{ticker}] Iniciando walk-forward validation para {model_type}...")

    # Carregar dados históricos
    X, y = prepare_dataset(ticker)
    df = X.copy()
    df["target"] = y

    # Ordenar por data
    df = df.sort_index()

    predictions = []
    actuals = []
    dates = []

    start_idx = train_window
    end_idx = len(df)

    while start_idx + test_window <= end_idx:
        # Dados de treino
        train_data = df.iloc[:start_idx]
        X_train = train_data.drop(columns=["target"])
        y_train = train_data["target"]

        # Dados de teste
        test_start = start_idx
        test_end = min(start_idx + test_window, end_idx)
        test_data = df.iloc[test_start:test_end]
        X_test = test_data.drop(columns=["target"])
        y_test = test_data["target"]

        # Treinar modelo (simplificado - em produção, treinar do zero)
        # Aqui assumimos que o modelo já foi treinado com dados históricos
        # Para backtesting real, seria necessário treinar o modelo em cada janela

        # Para simplificar, usar o modelo pré-treinado
        model = load_model(ticker, model_type)

        # Fazer previsões
        if model_type == "lstm":
            # Para LSTM, preparar sequências da última parte dos dados de treino + teste
            seq_length = 20
            if len(X_train) < seq_length:
                print("Dados insuficientes para LSTM")
                break

            # Usar scaler salvo
            from ml.models import StockLSTM
            import joblib
            import torch

            scaler_X_path = Path("ml/models") / f"{ticker}_lstm_scaler_X.pkl"
            scaler_y_path = Path("ml/models") / f"{ticker}_lstm_scaler_y.pkl"

            if not scaler_X_path.exists() or not scaler_y_path.exists():
                print("Scalers não encontrados para LSTM")
                break

            scaler_X = joblib.load(scaler_X_path)
            scaler_y = joblib.load(scaler_y_path)

            # Preparar sequência
            last_seq = X_test.iloc[:seq_length].values
            last_seq_scaled = scaler_X.transform(last_seq)
            last_seq_tensor = torch.tensor(last_seq_scaled, dtype=torch.float32).unsqueeze(0)

            model.eval()
            with torch.no_grad():
                pred_scaled = model(last_seq_tensor).item()
                pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]

            predictions.append(pred)
            actuals.append(y_test.iloc[0])  # Prever apenas o próximo dia
            dates.append(test_data.index[0])
        else:
            preds = model.predict(X_test)
            predictions.extend(preds)
            actuals.extend(y_test.values)
            dates.extend(test_data.index)

        start_idx += step

    # Calcular métricas
    mae = mean_absolute_error(actuals, predictions)
    rmse = mean_squared_error(actuals, predictions) ** 0.5

    print(".4f")
    print(".4f")

    return {
        "predictions": predictions,
        "actuals": actuals,
        "dates": dates,
        "mae": mae,
        "rmse": rmse,
    }


def main():
    parser = argparse.ArgumentParser(description="Backtesting de modelos de previsão")
    parser.add_argument("--ticker", default="AAPL", help="Ticker para backtesting")
    parser.add_argument("--model-type", default="random_forest", choices=["linear", "random_forest", "lstm"], help="Tipo de modelo")
    parser.add_argument("--train-window", type=int, default=252, help="Janela de treino (dias)")
    parser.add_argument("--test-window", type=int, default=21, help="Janela de teste (dias)")
    parser.add_argument("--step", type=int, default=21, help="Passo de avanço (dias)")

    args = parser.parse_args()

    result = walk_forward_validation(
        ticker=args.ticker,
        model_type=args.model_type,
        train_window=args.train_window,
        test_window=args.test_window,
        step=args.step,
    )

    print(f"\nBacktesting concluído para {args.ticker} com {args.model_type}")
    print(f"Total de previsões: {len(result['predictions'])}")


if __name__ == "__main__":
    main()