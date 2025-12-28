# Stock Forecast Service

[![CI](https://github.com/gabiRioRange/stock-forecast-service/actions/workflows/ci.yml/badge.svg)](https://github.com/gabiRioRange/stock-forecast-service/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Serviço de previsão de preços de ações usando Machine Learning com FastAPI, scikit-learn e PyTorch.

## 🚀 Funcionalidades

- **Previsão de Preços**: Random Forest e LSTM para previsão de preços de ações
- **API REST**: Endpoints para forecast, histórico e health check
- **Dashboard Interativo**: Interface Streamlit para visualização
- **Backtesting**: Validação walk-forward dos modelos
- **Cache Inteligente**: Cache de modelos para performance
- **Features Técnicas**: RSI, MACD, médias móveis, volatilidade

## 📊 Modelos Disponíveis

- **Random Forest**: Modelo ensemble tradicional
- **LSTM**: Rede Neural Recorrente para séries temporais

## 🛠️ Tecnologias

- **Backend**: FastAPI, Uvicorn
- **ML**: scikit-learn, PyTorch, pandas, numpy
- **Dados**: yfinance
- **Frontend**: Streamlit, Plotly, Matplotlib
- **Cache**: cachetools
- **Testes**: pytest
- **CI/CD**: GitHub Actions

## 💻 Requisitos do Sistema

- **Python**: 3.9+
- **RAM**: 4GB+ recomendado para treinamento
- **Espaço**: 2GB+ para modelos e dados
- **Internet**: Conexão para baixar dados do Yahoo Finance

## 🚀 Instalação

### Método Rápido (Recomendado)

1. Clone o repositório:
```bash
git clone <repository-url>
cd stock-forecast-service
```

2. Execute o setup automatizado:
```bash
python setup.py
```

Este comando irá:
- ✅ Instalar todas as dependências
- ✅ Executar testes
- ✅ Oferecer opção de treinar modelos
- ✅ Mostrar próximos passos

### Instalação Manual

1. Clone o repositório:
```bash
git clone <repository-url>
cd stock-forecast-service
```

2. Crie e ative o ambiente virtual:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 📈 Uso

### 1. Treinar Modelos

```bash
# Treinar Random Forest
python -m ml.train --tickers AAPL --model-type random_forest

# Treinar LSTM
python -m ml.train --tickers AAPL --model-type lstm
```

### 2. Iniciar API

```bash
python -m app.main
```

A API estará disponível em: http://localhost:8000

### 3. Executar Dashboard

```bash
streamlit run dashboard.py
```

O dashboard estará disponível em: http://localhost:8501

### 4. Backtesting

```bash
# Backtesting Random Forest
python -m ml.backtest --ticker AAPL --model-type random_forest

# Backtesting LSTM
python -m ml.backtest --ticker AAPL --model-type lstm
```

## 📡 API Endpoints

### Health Check
```
GET /health
```

### Lista de Assets
```
GET /assets
```

### Histórico de Preços
```
GET /history/{ticker}?start_date=2023-01-01&end_date=2024-01-01
```

### Previsão
```
GET /forecast/{ticker}?model_type=random_forest&horizon_days=7
```

## 🧪 Testes

Executar todos os testes:
```bash
pytest tests/
```

Testes específicos:
```bash
pytest tests/test_forecast.py -v
pytest tests/test_health.py -v
```

## � Scripts Disponíveis

- `setup.py` - Setup automatizado do projeto
- `run_api.py` - Executa apenas a API FastAPI
- `run_server.py` - Executa API + Dashboard simultaneamente
- `dashboard.py` - Interface Streamlit standalone

## 🏗️ Arquitetura

```
stock-forecast-service/
├── app/
│   ├── main.py          # FastAPI app
│   ├── routes/          # API endpoints
│   ├── services/        # Business logic
│   └── schemas.py       # Pydantic models
├── ml/
│   ├── models.py        # ML models & training
│   ├── data_prep.py     # Data processing
│   ├── train.py         # Training CLI
│   └── backtest.py      # Backtesting
├── tests/               # Unit tests
├── dashboard.py         # Streamlit dashboard
└── requirements.txt     # Dependencies
```

## 🔧 Desenvolvimento

### CI/CD

Este projeto utiliza GitHub Actions para CI/CD automático:

- ✅ **Testes Automáticos**: Executados em push/PR para `main` e `master`
- ✅ **Verificação de Qualidade**: Ambiente Python 3.11
- ✅ **Dependências**: Instalação automática via `requirements.txt`

### Adicionar Novo Modelo

1. Implementar função de treinamento em `ml/models.py`
2. Adicionar loader em `load_model()`
3. Atualizar endpoint `/forecast` para suportar novo tipo
4. Adicionar testes

### Adicionar Nova Feature

1. Modificar `data_prep.py` para incluir nova feature
2. Retreinar modelos
3. Atualizar documentação

## 📈 Melhorias Futuras

- [ ] Suporte a mais indicadores técnicos
- [ ] Modelos ensemble (Random Forest + LSTM)
- [ ] Previsão multi-step
- [ ] Interface web mais avançada
- [ ] Deploy em nuvem (Heroku, Railway, etc.)
- [ ] Rate limiting e autenticação
- [ ] Notificações em tempo real
- [ ] Suporte a criptomoedas
- [ ] API de webhooks
- [ ] Dashboard com mais gráficos interativos

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.

## 📞 Contato

Para dúvidas ou sugestões, abra uma issue no GitHub.