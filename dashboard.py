import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

# Configuração da página
st.set_page_config(
    page_title="Stock Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Forecast Dashboard")
st.markdown("Previsão de preços de ações usando Machine Learning")

# Verificar status da API
API_BASE_URL = "http://localhost:8000"

try:
    health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
    if health_response.status_code == 200:
        st.success("✅ API Online - Sistema funcionando!")
        api_online = True
    else:
        st.error(f"❌ API Offline - Status: {health_response.status_code}")
        api_online = False
        st.stop()
except Exception as e:
    st.error(f"⚠️ Erro ao conectar com API: {e}")
    st.info("💡 Execute: `python -m app.main` para iniciar a API")
    api_online = False
    st.stop()

# Sidebar
st.sidebar.header("⚙️ Configurações")

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"]
ticker = st.sidebar.selectbox("📊 Ticker", tickers, index=0)

model_type = st.sidebar.selectbox(
    "🤖 Modelo",
    ["random_forest", "lstm"],
    index=0
)

horizon_days = st.sidebar.slider("📅 Dias", 1, 7, 1)

# Botão para carregar dados
if st.sidebar.button("🔄 Carregar Dados", type="primary"):
    with st.spinner("Carregando dados..."):
        try:
            # Histórico
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)

            history_resp = requests.get(
                f"{API_BASE_URL}/history/{ticker}",
                params={
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d")
                },
                timeout=10
            )
            history_resp.raise_for_status()
            history_data = history_resp.json()

            # Forecast
            forecast_resp = requests.get(
                f"{API_BASE_URL}/forecast/{ticker}",
                params={"model_type": model_type, "horizon_days": horizon_days},
                timeout=10
            )
            forecast_resp.raise_for_status()
            forecast_data = forecast_resp.json()

            # Salvar dados
            st.session_state.history = history_data
            st.session_state.forecast = forecast_data
            st.session_state.ticker = ticker
            st.session_state.model_type = model_type

            st.success("✅ Dados carregados!")

        except Exception as e:
            st.error(f"Erro: {e}")

# Conteúdo principal
if 'history' in st.session_state and 'forecast' in st.session_state:
    history = st.session_state.history
    forecast = st.session_state.forecast

    df_history = pd.DataFrame(history['items'])
    df_history['date'] = pd.to_datetime(df_history['date'])
    df_history = df_history.sort_values('date')

    df_forecast = pd.DataFrame(forecast['predictions'])
    df_forecast['date'] = pd.to_datetime(df_forecast['date'])

    # Métricas
    col1, col2, col3 = st.columns(3)

    current_price = df_history['close'].iloc[-1]
    predicted_price = df_forecast['close'].iloc[0]
    delta = predicted_price - current_price
    delta_percent = (delta / current_price) * 100

    with col1:
        st.metric("💰 Preço Atual", f"${current_price:.2f}")

    with col2:
        st.metric(
            "🔮 Previsão",
            f"${predicted_price:.2f}",
            f"{delta:+.2f} ({delta_percent:+.2f}%)"
        )

    with col3:
        st.metric("📈 Modelo", model_type.upper())

    # Gráfico simples
    st.subheader(f"📊 {ticker} - Histórico + Previsão")

    # Usar matplotlib ao invés de plotly para evitar problemas
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df_history['date'], df_history['close'], label='Histórico', color='blue', linewidth=2)
    ax.plot(df_forecast['date'], df_forecast['close'], label='Previsão', color='red', linestyle='--', marker='o', linewidth=2, markersize=6)

    ax.set_title(f'Preço de Fechamento - {ticker}')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço (USD)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Formatar datas
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    st.pyplot(fig)

    # Tabela
    st.subheader("📋 Previsões")

    df_display = df_forecast.copy()
    df_display['date'] = df_display['date'].dt.strftime('%d/%m/%Y')
    df_display['close'] = df_display['close'].round(2)
    df_display = df_display[['date', 'close']]
    df_display.columns = ['Data', 'Preço Previsto (USD)']

    st.dataframe(df_display, use_container_width=True)

else:
    st.info("👆 Clique em 'Carregar Dados' para visualizar as previsões")

    st.markdown("""
    ### 📊 Performance dos Modelos
    | Modelo | MAE | RMSE |
    |--------|-----|------|
    | Random Forest | 0.0228 | 0.0294 |
    | LSTM | 0.0225 | 0.0288 |
    """)

# Footer
st.markdown("---")
st.markdown("*Dashboard - Stock Forecast Service*")