import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date
import plotly.graph_objects as go
from utils.savePortfolio import save_risk_metrics


# ---------- ACCESS GUARD ----------
if st.session_state.disable:
    st.info("Please enter your input on the Home page and try again.")
    st.stop()


# ---------- PORTFOLIO SETUP ----------
portfolio = st.session_state.portfolio
weights_dict = {}

tickers = portfolio["Ticker"].tolist()
values = portfolio["Value"].tolist()

total_value = sum(values)
for ticker, value in zip(tickers, values):
    weights_dict[ticker] = value / total_value

portfolio["Weight"] = portfolio["Ticker"].map(weights_dict)

weights = np.array(list(weights_dict.values()))

st.session_state["weights"] = weights
st.session_state["weights_dict"] = weights_dict
st.session_state["values"] = values
st.session_state["tickers"] = tickers


# ---------- TICKER NORMALIZATION (INDIA FIX) ----------
def normalize_ticker(t):
    indian = {
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN",
        "ITC", "LT", "AXISBANK", "KOTAKBANK", "BHARTIARTL",
        "HINDUNILVR", "BAJFINANCE", "MARUTI", "TATAMOTORS"
    }
    return f"{t}.NS" if t in indian else t


yf_tickers = [normalize_ticker(t) for t in tickers]


# ---------- DOWNLOAD STOCK DATA ----------
stocks = yf.download(yf_tickers, period="5y", group_by="ticker")

if stocks.empty:
    st.error("No stock data available. Please check ticker symbols.")
    st.stop()


# ---------- HANDLE MULTIINDEX ----------
def extract_close(df):
    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(1):
            return df.xs("Adj Close", level=1, axis=1)
        else:
            return df.xs("Close", level=1, axis=1)
    return df["Adj Close"] if "Adj Close" in df.columns else df["Close"]


stocks = extract_close(stocks)

if stocks.empty:
    st.error("Stock price data unavailable after processing.")
    st.stop()


# ---------- RETURNS & DRAWDOWN ----------
returns = stocks.pct_change().dropna()

if returns.empty:
    st.error("Not enough historical data to compute returns.")
    st.stop()

portfolio_returns = np.dot(returns.mean() * 252, weights)
portfolio_sd = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

cumulative = (1 + returns).cumprod()
rolling_max = cumulative.expanding().max()
drawdowns = cumulative / rolling_max - 1
max_drawdown = drawdowns.min().min()


# ---------- STORE SESSION ----------
st.session_state["stocks"] = stocks
st.session_state["returns"] = returns


# ---------- UI CONTROLS ----------
c1, c2, c3 = st.columns([2, 2, 1])

with c1:
    frequency = st.selectbox("Frequency of return", ["Daily", "Weekly", "Monthly", "Yearly"])

with c2:
    t = st.selectbox("Select ticker", tickers)

with c3:
    st.page_link("Pages/3_Risk Analysis.py", label="Go to Risk Analysis →")


st.title(f"{t} Summary")


# ---------- FREQUENCY RETURNS ----------
if frequency == "Daily":
    f = stocks.pct_change()
elif frequency == "Weekly":
    f = stocks.resample("W-FRI").last().pct_change()
elif frequency == "Monthly":
    f = stocks.resample("M").last().pct_change()
else:
    f = stocks.resample("Y").last().pct_change()

f = f.dropna()


# ---------- RISK TABLE ----------
annual_volatility = (returns.std() * np.sqrt(252) * 100).round(2)
risk = annual_volatility.to_frame(name="Volatility")
st.session_state["risk"] = risk


# ---------- CHARTS ----------
col1, col2 = st.columns([3, 2])

with col1:
    tab1, tab2 = st.tabs(["Historical Stock Price", "Live Price"])

    with tab1:
        st.line_chart(stocks[t], height=350)

    with tab2:
        yday = stocks.index[-2]
        ohlc = yf.Ticker(normalize_ticker(t)).history(
            start=yday, end=date.today(), interval="2m"
        )

        if not ohlc.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=ohlc.index,
                open=ohlc["Open"],
                high=ohlc["High"],
                low=ohlc["Low"],
                close=ohlc["Close"]
            )])
            fig.update_layout(height=400)
            st.plotly_chart(fig)
        else:
            st.info("Live data not available.")


# ---------- NEWS & METRICS ----------
with col2:
    st.subheader(f"{t} News")
    news = yf.Ticker(normalize_ticker(t)).news or []

    shown = 0
    for article in news:
        if article.get("title") and article.get("link"):
            st.markdown(f"[{article['title']}]({article['link']})")
            shown += 1
        if shown == 3:
            break

    pct = f[t].iloc[-1] * 100 if not f.empty else 0

    st.metric("Current Stock Price", round(stocks[t].iloc[-1], 2), f"{pct:.2f}%")
    st.metric("Portfolio Weight", round(weights_dict[t], 2))
    st.metric("Annualised Volatility", f"{annual_volatility[t]}%")
    st.metric("Maximum Drawdown", f"{round(drawdowns[t].min() * 100, 2)}%")


# ---------- RETURNS CHART ----------
st.write(f"{frequency} returns (Last 5 years)")
st.line_chart(f[t], height=140)


# ---------- SAVE (LOCAL ONLY, COLLEGE MODE) ----------
save_risk_metrics(
    portfolio=portfolio,
    stocks=stocks,
    weights_dict=weights_dict,
    returns=returns,
    riskfree=0.05,
    values=values,
    max_drawdown=max_drawdown,
    portfolio_returns=portfolio_returns,
    portfolio_sd=portfolio_sd,
    tickers=tickers
)
