from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


MODELS_DIR = Path("ml") / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelMetrics:
    mae: float
    rmse: float


def train_sklearn_regressor(
    X,
    y,
    ticker: str,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Treina modelo sklearn (linear ou random_forest) e salva .pkl."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=200, random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    model_path = MODELS_DIR / f"{ticker}_{model_type}.pkl"
    joblib.dump(model, model_path)

    return {
        "model_path": str(model_path),
        "metrics": ModelMetrics(mae=mae, rmse=rmse),
    }


def load_model(ticker: str, model_type: str = "random_forest"):
    """Carrega modelo salvo em .pkl ou .pt."""
    if model_type in ["linear", "random_forest"]:
        model_path = MODELS_DIR / f"{ticker}_{model_type}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return joblib.load(model_path)
    elif model_type == "lstm":
        model_path = MODELS_DIR / f"{ticker}_{model_type}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = StockLSTM(input_size=8, hidden_size=50, num_layers=2, output_size=1)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


class StockLSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=50, num_layers=2, output_size=1):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def create_sequences(data, seq_length=20):
    """Cria sequências para LSTM."""
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length, 0]  # Prever close
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)


def train_lstm_regressor(
    X,
    y,
    ticker: str,
    seq_length: int = 20,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> Dict[str, Any]:
    """Treina modelo LSTM e salva .pt."""
    # Normalizar dados
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # Criar sequências apenas com features X
    X_seq, y_seq = create_sequences(X_scaled, seq_length)

    if len(X_seq) == 0:
        raise ValueError("Not enough data for sequences")

    # Split
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # DataLoader
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Modelo
    model = StockLSTM(input_size=X.shape[1], hidden_size=50, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Treino
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

    # Avaliação
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            y_pred.extend(outputs.squeeze().numpy())
            y_true.extend(targets.numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    # Salvar
    model_path = MODELS_DIR / f"{ticker}_lstm.pt"
    torch.save(model.state_dict(), model_path)

    # Salvar scalers
    scaler_X_path = MODELS_DIR / f"{ticker}_lstm_scaler_X.pkl"
    scaler_y_path = MODELS_DIR / f"{ticker}_lstm_scaler_y.pkl"
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)

    return {
        "model_path": str(model_path),
        "scaler_X_path": str(scaler_X_path),
        "scaler_y_path": str(scaler_y_path),
        "metrics": ModelMetrics(mae=mae, rmse=rmse),
    }
