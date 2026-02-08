import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go


# ---------- PAGE CONFIG ----------
st.set_page_config(layout="wide")

st.markdown(
    """
<style>
.header-card {
    padding: 18px;
    border-radius: 14px;
    background: linear-gradient(135deg,#020617,#020617);
    border: 1px solid #1e293b;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------- HEADER ----------
st.markdown(
    """
<div class="header-card">
<h2>🧠 LSTM Stock Price Prediction</h2>
<p>AI model forecasting the next closing price using sequential deep learning.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ---------- LOAD MODEL ----------
@st.cache_resource
def load_lstm():
    return load_model("models/lstm_model.h5", compile=False)


model = load_lstm()


# ---------- SCALING FUNCTIONS ----------
def minmax_scale(data, mn, mx):
    return (data - mn) / (mx - mn + 1e-9)


def inverse_scale(val, mn, mx):
    return val * (mx - mn + 1e-9) + mn


def predict_next_price(close_prices):
    mn = np.min(close_prices)
    mx = np.max(close_prices)

    scaled = minmax_scale(close_prices, mn, mx)

    if len(scaled) < 5:
        return None

    seq = scaled[-5:]
    seq = np.array(seq).reshape(1, 5, 1)

    pred_scaled = model.predict(seq, verbose=0)[0][0]
    return float(inverse_scale(pred_scaled, mn, mx))


# ---------- INPUT ----------
st.markdown("### 📥 Enter Stock Symbol")

col1, col2 = st.columns([3, 1])

with col1:
    symbol = st.text_input(
        "Stock ticker",
        placeholder="Examples: AAPL, TSLA, RELIANCE.NS",
    )

with col2:
    predict_button = st.button("Run Prediction")


# ---------- DATA + PREDICTION ----------
if symbol and predict_button:

    with st.spinner("Fetching stock data and running LSTM inference..."):
        df = yf.download(symbol, period="1y", progress=False)

    # ----- VALIDATION -----
    if df is None or df.empty:
        st.error("No market data found. Check ticker symbol.")
        st.stop()

    # Handle all yfinance formats safely
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(1):
            df = df.xs("Close", level=1, axis=1)
        else:
            df = df.xs(df.columns.get_level_values(1)[0], level=1, axis=1)

    if "Close" not in df.columns:
        st.error("Close price not available for this ticker.")
        st.stop()

    # Ensure numeric 1-D float array
    close_prices = df["Close"].astype(float).values.flatten()

    if len(close_prices) < 10:
        st.error("Not enough historical data for prediction.")
        st.stop()

    prediction = predict_next_price(close_prices)

    if prediction is None:
        st.error("Prediction failed due to insufficient sequence.")
        st.stop()

    # ---------- METRICS ----------
    st.markdown("### 📊 Prediction Output")

    m1, m2 = st.columns(2)

    last_price = float(close_prices[-1])
    predicted_price = float(prediction)
    change = predicted_price - last_price

    m1.metric("Last Close", f"{round(last_price,2)}")

    m2.metric(
        "Predicted Next Close",
        f"{round(predicted_price,2)}",
        delta=round(change, 2),
    )

    # ---------- CHART ----------
    st.markdown("### 📈 Price Trend & Forecast")

    fig = go.Figure()

    # Historical trend
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            line=dict(width=3),
            name="Historical Price",
        )
    )

    # Prediction marker
    fig.add_trace(
        go.Scatter(
            x=[df.index[-1]],
            y=[predicted_price],
            mode="markers+text",
            marker=dict(size=14),
            text=["Prediction"],
            textposition="top center",
            name="Predicted Price",
        )
    )

    fig.update_layout(
        height=500,
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------- MODEL INFO ----------
    with st.expander("ℹ️ Model Explanation"):
        st.write(
            "This prediction uses an LSTM neural network trained on sequential closing prices."
        )
        st.write(
            "The model evaluates the last 5 timesteps to forecast the next price movement."
        )
        st.write(
            "Predictions are probabilistic and should not be used as financial advice."
        )
