import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import json
import os
from ai.sentiment_engine import analyze_sentiment
from data.news_fetcher import get_stock_news


# ---------- PAGE CONFIG ----------
st.set_page_config(layout="wide")

st.markdown(
    """
<style>
.header-card {
    padding: 20px;
    border-radius: 14px;
    background: linear-gradient(135deg,#020617,#0f172a);
    border: 1px solid #1e293b;
}
.metric-box {
    background:#020617;
    padding:12px;
    border-radius:10px;
    border:1px solid #1e293b;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------- HEADER ----------
st.markdown(
    """
<div class="header-card">
<h2>🤖 AI Multi-Model Stock Prediction</h2>
<p>LSTM + ARIMA + Prophet + News Sentiment Engine</p>
</div>
""",
    unsafe_allow_html=True,
)


# ---------- LOAD LSTM ----------
@st.cache_resource
def load_lstm():
    return load_model("models/lstm_model.h5", compile=False)


lstm_model = load_lstm()


# ---------- SAFE CLOSE EXTRACTION ----------
def extract_close(df):

    if isinstance(df.columns, pd.MultiIndex):

        level_vals = df.columns.get_level_values(-1)

        if "Close" in level_vals:
            return df.xs("Close", level=-1, axis=1)

        elif "Adj Close" in level_vals:
            return df.xs("Adj Close", level=-1, axis=1)

        else:
            return df.select_dtypes(include=[np.number]).iloc[:, 0]

    else:
        if "Close" in df.columns:
            return df["Close"]

        elif "Adj Close" in df.columns:
            return df["Adj Close"]

        else:
            return df.select_dtypes(include=[np.number]).iloc[:, 0]


# ---------- LSTM ----------
def predict_lstm(close_prices):
    close_prices = np.array(close_prices).astype(float).flatten()

    mn, mx = np.min(close_prices), np.max(close_prices)

    if mx - mn == 0:
        return float(close_prices[-1])

    scaled = (close_prices - mn) / (mx - mn + 1e-9)

    if len(scaled) < 5:
        return float(close_prices[-1])

    seq = scaled[-5:].reshape(1, 5, 1)
    pred_scaled = lstm_model.predict(seq, verbose=0)[0][0]
    return float(pred_scaled * (mx - mn) + mn)


# ---------- INPUT ----------
st.markdown("### 📥 Enter Stock Symbol")

col1, col2 = st.columns([3, 1])

with col1:
    symbol = st.text_input("Stock ticker", "AAPL")

with col2:
    run = st.button("Run AI Prediction")


# ---------- MAIN EXECUTION ----------
if symbol and run:

    # -------- FETCH DATA --------
    with st.spinner("Fetching market data..."):
        df = yf.download(symbol, period="2y", progress=False)

    if df is None or df.empty:
        st.error("No market data found.")
        st.stop()

    close = extract_close(df)
    close = close.astype(float).dropna()

    if len(close) < 60:
        st.warning("Not enough historical data.")
        st.stop()

    # =============================
    # SENTIMENT
    # =============================
    news = get_stock_news(symbol)
    sentiment_score = analyze_sentiment(news) if news else 0

    st.markdown("### 📰 Market Sentiment")

    if sentiment_score > 0.2:
        st.success(f"Positive ({round(sentiment_score,3)})")
    elif sentiment_score < -0.2:
        st.error(f"Negative ({round(sentiment_score,3)})")
    else:
        st.warning(f"Neutral ({round(sentiment_score,3)})")

    # =============================
    # TRAIN / TEST
    # =============================
    train = close[:-30]
    test = close[-30:]
    actual = float(test.iloc[0])

    # ---------- LSTM ----------
    lstm_pred = predict_lstm(train.values)

    # ---------- ARIMA ----------
    try:
        arima = ARIMA(train, order=(5, 1, 0)).fit()
        arima_pred = float(arima.forecast(steps=1).iloc[0])
    except:
        arima_pred = float(train.iloc[-1])

    # ---------- PROPHET ----------
    try:
        prophet_df = train.reset_index()
        prophet_df.columns = ["ds", "y"]
        prophet = Prophet(daily_seasonality=True)
        prophet.fit(prophet_df)
        future = prophet.make_future_dataframe(periods=1)
        forecast = prophet.predict(future)
        prophet_pred = float(forecast["yhat"].iloc[-1])
    except:
        prophet_pred = float(train.iloc[-1])

    # =============================
    # METRICS
    # =============================
    st.markdown("### 📊 Prediction Comparison")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("LSTM", round(lstm_pred, 2))
    m2.metric("ARIMA", round(arima_pred, 2))
    m3.metric("Prophet", round(prophet_pred, 2))
    m4.metric("Actual", round(actual, 2))

    # =============================
    # RMSE
    # =============================
    lstm_rmse = np.sqrt(mean_squared_error([actual], [lstm_pred]))
    arima_rmse = np.sqrt(mean_squared_error([actual], [arima_pred]))
    prophet_rmse = np.sqrt(mean_squared_error([actual], [prophet_pred]))

    st.markdown("### 📉 Model Accuracy")

    r1, r2, r3 = st.columns(3)
    r1.metric("LSTM RMSE", round(lstm_rmse, 4))
    r2.metric("ARIMA RMSE", round(arima_rmse, 4))
    r3.metric("Prophet RMSE", round(prophet_rmse, 4))

    # =============================
    # SAVE FOR AI AGENT
    # =============================
    predictions_path = os.path.join("utils", "model_predictions.json")

    if os.path.exists(predictions_path):
        with open(predictions_path, "r") as f:
            all_preds = json.load(f)
    else:
        all_preds = {}

    all_preds[symbol] = {
        "lstm": float(lstm_pred),
        "arima": float(arima_pred),
        "prophet": float(prophet_pred),
        "actual": float(actual),
        "sentiment": float(sentiment_score),
    }

    with open(predictions_path, "w") as f:
        json.dump(all_preds, f, indent=4)

    # =============================
    # INTERACTIVE TREND CHART
    # =============================
    st.markdown("### 📈 Historical Trend + Model Predictions")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=close.index, y=close, mode="lines", line=dict(width=3), name="Historical"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[close.index[-1]],
            y=[lstm_pred],
            mode="markers",
            marker=dict(size=12),
            name="LSTM",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[close.index[-1]],
            y=[arima_pred],
            mode="markers",
            marker=dict(size=12),
            name="ARIMA",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[close.index[-1]],
            y=[prophet_pred],
            mode="markers",
            marker=dict(size=12),
            name="Prophet",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=520,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)
