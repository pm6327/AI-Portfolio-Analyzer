import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import json
import os
from ai.sentiment_engine import analyze_sentiment
from data.news_fetcher import get_stock_news
from TTF.state import init_session_state


# -------- Load LSTM --------
@st.cache_resource
def load_lstm():
    return load_model("models/lstm_model.h5", compile=False)


lstm_model = load_lstm()


def predict_lstm(close_prices):
    mn = close_prices.min()
    mx = close_prices.max()
    scaled = (close_prices - mn) / (mx - mn)
    seq = scaled[-5:]
    seq = np.array(seq).reshape(1, 5, 1)
    pred_scaled = lstm_model.predict(seq, verbose=0)[0][0]
    return pred_scaled * (mx - mn) + mn


# -------- UI --------
symbol = st.text_input("Enter Stock Symbol", "AAPL")

if symbol:

    # -------- Fetch stock data --------
    df = yf.download(symbol, period="2y")
    df = df.dropna()

    if df.empty:
        st.warning("No stock data found.")
        st.stop()

    close = df["Close"]

    # =============================
    # NEWS SENTIMENT
    # =============================
    news = get_stock_news(symbol)

    if news:
        sentiment_score = analyze_sentiment(news)
    else:
        sentiment_score = 0

    st.subheader("Market Sentiment")

    if sentiment_score > 0.2:
        st.success(f"Positive ({round(sentiment_score,3)})")
    elif sentiment_score < -0.2:
        st.error(f"Negative ({round(sentiment_score,3)})")
    else:
        st.warning(f"Neutral ({round(sentiment_score,3)})")

    # -------- split train-test --------
    train = close[:-30]
    test = close[-30:]

    # -------- LSTM prediction --------
    lstm_pred = predict_lstm(train.values)

    # -------- ARIMA --------
    arima_model = ARIMA(train, order=(5, 1, 0))
    arima_fit = arima_model.fit()
    arima_pred = arima_fit.forecast(steps=1).iloc[0]

    # -------- Prophet --------
    prophet_df = train.reset_index()
    prophet_df.columns = ["ds", "y"]

    prophet_model = Prophet()
    prophet_model.fit(prophet_df)

    future = prophet_model.make_future_dataframe(periods=1)
    forecast = prophet_model.predict(future)
    prophet_pred = forecast["yhat"].iloc[-1]

    # -------- Actual last value --------
    actual = test.iloc[0]

    # -------- Metrics --------
    lstm_rmse = np.sqrt(mean_squared_error([actual], [lstm_pred]))
    arima_rmse = np.sqrt(mean_squared_error([actual], [arima_pred]))
    prophet_rmse = np.sqrt(mean_squared_error([actual], [prophet_pred]))

    # -------- Display predictions --------
    st.subheader("Prediction Comparison")

    st.write(f"LSTM Prediction: {round(lstm_pred,2)}")
    st.write(f"ARIMA Prediction: {round(arima_pred,2)}")
    st.write(f"Prophet Prediction: {round(prophet_pred,2)}")
    st.write(f"Actual Price: {round(actual,2)}")

    # =============================
    # Save predictions for AI agent
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

    # -------- RMSE --------
    st.subheader("RMSE Comparison")

    st.write(f"LSTM RMSE: {lstm_rmse}")
    st.write(f"ARIMA RMSE: {arima_rmse}")
    st.write(f"Prophet RMSE: {prophet_rmse}")

    # -------- Plot --------
    fig, ax = plt.subplots()
    ax.plot(close.values, label="Historical")

    ax.scatter(len(close) - 1, lstm_pred, label="LSTM")
    ax.scatter(len(close) - 1, arima_pred, label="ARIMA")
    ax.scatter(len(close) - 1, prophet_pred, label="Prophet")

    ax.legend()
    st.pyplot(fig)
