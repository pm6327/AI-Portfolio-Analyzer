import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model


st.title("LSTM Model Prediction")


# load model once
@st.cache_resource
def load_lstm():
    return load_model("models/lstm_model.h5", compile=False)


model = load_lstm()


def minmax_scale(data, mn, mx):
    return (data - mn) / (mx - mn)


def inverse_scale(val, mn, mx):
    return val * (mx - mn) + mn


def predict_next_price(close_prices):
    mn = close_prices.min()
    mx = close_prices.max()

    scaled = minmax_scale(close_prices, mn, mx)
    seq = scaled[-5:]
    seq = np.array(seq).reshape(1, 5, 1)

    pred_scaled = model.predict(seq, verbose=0)[0][0]
    return inverse_scale(pred_scaled, mn, mx)


# ---- UI ----

symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, RELIANCE.NS)")

if symbol:
    df = yf.download(symbol, period="1y")

    if df.empty:
        st.error("Invalid symbol or data not found")
    else:
        close_prices = df["Close"].values

        prediction = predict_next_price(close_prices)

        st.subheader("Prediction Result")

        st.metric(label="Predicted Next Close Price", value=f"{round(prediction, 2)}")

        st.line_chart(df["Close"])

        st.write("LSTM predicted next closing price based on last 5 timesteps.")
