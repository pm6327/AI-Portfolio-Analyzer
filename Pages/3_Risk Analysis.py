import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
from utils.savePortfolio import save_risk_metrics


# ---------- PAGE CONFIG ----------
st.set_page_config(layout="wide")

st.markdown(
    """
<style>
.header-card {
    padding: 18px;
    border-radius: 14px;
    background: linear-gradient(135deg,#0f172a,#020617);
    border: 1px solid #1e293b;
}
.metric-card {
    padding: 14px;
    border-radius: 10px;
    background-color: #020617;
    border: 1px solid #1e293b;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------- ACCESS GUARD ----------
if st.session_state.disable:
    st.info("Please enter your input on the Home page and try again.")
    st.stop()


# ---------- HEADER ----------
h1, h2 = st.columns([4, 1], vertical_alignment="bottom")

with h1:
    st.markdown(
        """
    <div class="header-card">
    <h2>📉 Portfolio Risk Analysis</h2>
    <p>Advanced analytics on volatility, risk exposure and downside protection.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with h2:
    st.page_link("Pages/4_Optimal Portfolio.py", label="Optimise Portfolio →")


# ---------- SESSION DATA ----------
portfolio = st.session_state["portfolio"]
tickers = st.session_state["tickers"]
values = st.session_state["values"]
weights = st.session_state["weights"]
weights_dict = st.session_state["weights_dict"]
stocks = st.session_state["stocks"]
returns = st.session_state["returns"]
risk = st.session_state["risk"]

risk["items"] = ""
risk["norm_vol"] = risk["Volatility"] / risk["Volatility"].sum()
risk["weight"] = portfolio.set_index("Ticker")["Weight"]


# ---------- CALCULATIONS ----------
covariance = returns.cov()

portfolio_sd = np.sqrt(
    np.dot(np.array(weights).T, np.dot(covariance, np.array(weights)))
) * np.sqrt(252)

portfolio_returns = np.dot(weights, returns.mean() * 252)

daily_returns = returns.dot(weights)
cumulative_returns = (1 + daily_returns).cumprod()
running_max = cumulative_returns.expanding().max()
drawdown = (cumulative_returns - running_max) / running_max

max_drawdown = drawdown.min()
max_drawdown_value = max_drawdown * sum(values)

riskfree = 0.021
sharpe = (portfolio_returns - riskfree) / portfolio_sd


# ---------- SAVE ----------
save_risk_metrics(
    portfolio,
    stocks,
    weights_dict,
    returns,
    riskfree=riskfree,
    values=values,
    max_drawdown=max_drawdown,
    portfolio_returns=portfolio_returns,
    portfolio_sd=portfolio_sd,
    tickers=tickers,
)


# ---------- HERO METRICS ----------
st.markdown("### 📊 Portfolio Risk Snapshot")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Total Value", f"₹ {round(sum(values),2)}")
m2.metric("Annual Return", f"{round(portfolio_returns*100,2)}%")
m3.metric("Volatility", f"{round(portfolio_sd*100,2)}%")
m4.metric("Sharpe Ratio", round(sharpe, 2))


# ---------- DISTRIBUTION VISUALS ----------
st.markdown("### 📦 Allocation & Risk Composition")

fig1 = px.bar(
    risk,
    x="weight",
    y="items",
    color=risk.index,
    orientation="h",
    text_auto=".0%",
    height=280,
    color_discrete_sequence=px.colors.qualitative.G10,
    title="Portfolio Allocation",
)

fig1.update_yaxes(visible=False)
fig1.update_xaxes(visible=False)

fig2 = px.bar(
    risk,
    x="norm_vol",
    y="items",
    color=risk.index,
    orientation="h",
    text_auto=".0%",
    height=280,
    color_discrete_sequence=px.colors.qualitative.G10,
    title="Risk Contribution",
)

fig2.update_yaxes(visible=False)
fig2.update_xaxes(visible=False)

c1, c2 = st.columns(2)
c1.plotly_chart(fig1, use_container_width=True)
c2.plotly_chart(fig2, use_container_width=True)


# ---------- ADVANCED METRICS ----------
st.markdown("### 🧠 Advanced Risk Metrics")

metrics_tabs = ["Sharpe", "Sortino", "Drawdown", "Value at Risk"]
tab1, tab2, tab3, tab4 = st.tabs(metrics_tabs)


# ----- SHARPE TAB
with tab1:
    c1, c2 = st.columns([1, 3])
    c1.metric("Sharpe Ratio", round(sharpe, 2))
    c2.write(
        "Measures risk-adjusted returns by comparing portfolio excess return to volatility."
    )
    c2.write("Higher Sharpe = better performance per unit risk.")


# ----- SORTINO TAB
with tab2:
    downside_returns = returns.where(returns < 0)
    downside_deviation = np.sqrt(
        np.dot(np.array(weights).T, np.dot(downside_returns.cov(), np.array(weights)))
    ) * np.sqrt(252)

    sortino = (portfolio_returns - riskfree) / downside_deviation

    c1, c2 = st.columns([1, 3])
    c1.metric("Sortino Ratio", round(sortino, 2))
    c2.write("Focuses only on downside volatility instead of total volatility.")
    c2.write("Higher Sortino indicates efficient downside risk management.")


# ----- MAX DRAWDOWN TAB
with tab3:
    c1, c2 = st.columns([1, 3])
    c1.metric(
        "Maximum Drawdown",
        f"{round(max_drawdown*100,2)}%",
        delta=round(max_drawdown_value, 2),
    )
    c2.write("Largest peak-to-trough fall in portfolio value over the analysis period.")
    c2.write("Lower drawdown implies capital protection.")


# ----- VAR TAB
with tab4:
    confidence_level = 0.95
    z_score = norm.ppf(1 - confidence_level)

    var_relative = -(
        np.sum(returns * weights, axis=1).mean() * 10
        + z_score * portfolio_sd / np.sqrt(252) * np.sqrt(10)
    )

    var_absolute = sum(values) * var_relative

    c1, c2 = st.columns([1, 3])

    c1.metric(
        "10-day VaR (95%)",
        f"{round(var_relative*100,2)}%",
        delta=round(-var_absolute, 0),
    )

    c2.write("Estimates potential maximum loss over 10 days with 95% confidence.")
    c2.write("Used in institutional risk management frameworks.")
