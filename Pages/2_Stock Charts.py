import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date
import plotly.graph_objects as go
from utils.savePortfolio import save_risk_metrics


# ---------- PAGE CONFIG ----------
st.set_page_config(layout="wide")

st.markdown(
    """
<style>
.metric-card {
    background-color: #111827;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #1f2937;
}
.news-card {
    background-color: #0f172a;
    padding: 14px;
    border-radius: 10px;
    margin-bottom: 10px;
    border: 1px solid #1e293b;
}
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)


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


# ---------- TICKER NORMALIZATION ----------
def normalize_ticker(t):
    indian = {
        "RELIANCE",
        "TCS",
        "INFY",
        "HDFCBANK",
        "ICICIBANK",
        "SBIN",
        "ITC",
        "LT",
        "AXISBANK",
        "KOTAKBANK",
        "BHARTIARTL",
        "HINDUNILVR",
        "BAJFINANCE",
        "MARUTI",
        "TATAMOTORS",
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

st.session_state["stocks"] = stocks
st.session_state["returns"] = returns


# ---------- HERO DASHBOARD ----------
st.markdown("## 📊 Portfolio Overview")

hero1, hero2, hero3, hero4 = st.columns(4)

hero1.metric("Portfolio Return", f"{round(portfolio_returns*100,2)}%")
hero2.metric("Portfolio Risk (SD)", f"{round(portfolio_sd*100,2)}%")
hero3.metric("Max Drawdown", f"{round(max_drawdown*100,2)}%")
hero4.metric("Assets Tracked", len(tickers))


# ---------- CONTROL BAR ----------
st.markdown("### ⚙️ Controls")

ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])

with ctrl1:
    frequency = st.selectbox(
        "Return Frequency", ["Daily", "Weekly", "Monthly", "Yearly"]
    )

with ctrl2:
    t = st.selectbox("Select Stock", tickers)

with ctrl3:
    st.page_link("Pages/3_Risk Analysis.py", label="Open Risk Lab →")


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


# ---------- MAIN DASHBOARD AREA ----------
left, right = st.columns([3, 1.4])

with left:
    st.markdown("### 📈 Price Movement")

    tab1, tab2 = st.tabs(["Historical", "Live Market"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stocks.index, y=stocks[t], mode="lines", name=t))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        yday = stocks.index[-2]
        ohlc = yf.Ticker(normalize_ticker(t)).history(
            start=yday, end=date.today(), interval="2m"
        )

        if not ohlc.empty:
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=ohlc.index,
                        open=ohlc["Open"],
                        high=ohlc["High"],
                        low=ohlc["Low"],
                        close=ohlc["Close"],
                    )
                ]
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Live data not available.")


# ---------- RIGHT PANEL ----------
with right:
    st.markdown("### 📊 Stock Insights")

    pct = f[t].iloc[-1] * 100 if not f.empty else 0

    st.metric("Current Price", round(stocks[t].iloc[-1], 2), f"{pct:.2f}%")
    st.metric("Portfolio Weight", f"{round(weights_dict[t]*100,2)}%")
    st.metric("Volatility", f"{annual_volatility[t]}%")
    st.metric("Drawdown", f"{round(drawdowns[t].min()*100,2)}%")

    st.markdown("### 📰 Latest News")

    news = yf.Ticker(normalize_ticker(t)).news or []

    for article in news[:3]:
        if article.get("title") and article.get("link"):
            st.markdown(
                f"""
                <div class="news-card">
                <a href="{article['link']}" target="_blank">
                {article['title']}
                </a>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------- RETURNS TREND ----------
st.markdown("### 📉 Returns Trend")

fig = go.Figure()
fig.add_trace(go.Scatter(x=f.index, y=f[t], mode="lines", name="Returns"))
fig.update_layout(height=250)
st.plotly_chart(fig, use_container_width=True)


# ---------- SAVE METRICS ----------
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
    tickers=tickers,
)
