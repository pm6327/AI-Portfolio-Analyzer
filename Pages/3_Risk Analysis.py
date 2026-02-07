import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
from utils.savePortfolio import save_risk_metrics


if st.session_state.disable:
    st.info("Please enter your input on the Home page and try again.")
    st.stop()

c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
c1.title("Risk Analysis")
c2.page_link("Pages/4_Optimal Portfolio.py", label="Optimise Portfolio →")

st.session_state["Query"] = "Hi"

# ----- LOCAL VARIABLES -----
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


# ----- GENERAL CALCULATIONS -----
covariance = returns.cov()
portfolio_sd = np.sqrt(
    np.dot(np.array(weights).T, np.dot(covariance, np.array(weights)))
) * np.sqrt(252)
portfolio_returns = np.dot(weights, returns.mean() * 252)

# ----- MAXIMUM DRAWDOWN -----
daily_returns = returns.dot(weights)  # Daily portfolio returns
cumulative_returns = (1 + daily_returns).cumprod()  # Cumulative returns
running_max = cumulative_returns.expanding().max()  # Running maximum
drawdown = (cumulative_returns - running_max) / running_max  # Drawdown series
max_drawdown = drawdown.min()
max_drawdown_value = max_drawdown * sum(values)

save_risk_metrics(
    portfolio,
    stocks,
    weights_dict,
    returns,
    riskfree=0.021,
    values=values,
    max_drawdown=max_drawdown,
    portfolio_returns=portfolio_returns,
    portfolio_sd=portfolio_sd,
    tickers=tickers,
)


# ----- PORTFOLIO SHARPE RATIO -----
riskfree = 0.021

sharpe = (portfolio_returns - riskfree) / portfolio_sd

portfolio["items"] = ""
portfolio["Weight"] = portfolio["Ticker"].map(weights_dict)


# ----- To show portfolio distribution
fig1 = px.bar(
    risk,
    x="weight",
    y="items",
    color=risk.index,
    orientation="h",
    barmode="stack",
    text_auto=".0%",
    height=250,
    color_discrete_sequence=px.colors.qualitative.G10,
    title="Portfolio Distribution",
)
fig1.update_yaxes(visible=False, showticklabels=False)
fig1.update_xaxes(visible=False, showticklabels=False)
fig1.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="right", x=0.7),
    margin=dict(t=40, b=100),
)

# ----- To show risk distribution
fig2 = px.bar(
    risk,
    x="norm_vol",
    y=risk["items"],
    color=risk.index,
    orientation="h",
    barmode="stack",
    text_auto=".0%",
    height=250,
    color_discrete_sequence=px.colors.qualitative.G10,
    title="Risk Distribution",
)
fig2.update_yaxes(visible=False, showticklabels=False)
fig2.update_xaxes(visible=False, showticklabels=False)
fig2.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="right", x=0.7),
    margin=dict(t=40, b=100),
)


# ----- METRICS
col1, col2, col3 = st.columns(3, vertical_alignment="center")
with col1:
    st.metric("Total Portfolio Value", value="$" + str(sum(values)))
with col2:
    st.metric("Annualised Returns", value=str(round(portfolio_returns * 100, 2)) + "%")
with col3:
    st.metric("Annualised Volatility", value=str(round(portfolio_sd * 100, 2)) + "%")

col1, col2, col3 = st.columns([1, 0.2, 1], vertical_alignment="center")
col1.plotly_chart(fig1)
col3.plotly_chart(fig2)

metrics = ["Sharpe Ratio", "Sortino Ratio", "Maximum Drawdown", "Value-at-Risk"]
tab1, tab2, tab3, tab4 = st.tabs(metrics)
with tab1:
    col1, col2, col3 = st.columns([0.2, 1, 3])
    query = "Sharpe Ratio"
    with col2:
        st.metric(query, value=round(sharpe, 2))
    with col3:
        st.write("The **Sharpe Ratio** is a measure of risk-adjusted return.")
        st.write(
            "It is calculated by dividing excess return by portfolio's overall risk"
        )
        st.write("A higher Sharpe Ratio indicates better risk-adjusted performance.")
        st.write("Sharpe Ratio > 1 is generally a good investment")
with tab2:
    mar = 0  # For Sortino, we generally use 0 as MAR (Minimum Acceptable Return)
    # Calculate downside returns
    downside_returns = returns.where(returns < 0)
    # Calculate downside deviation (standard deviation of downside returns)
    downside_deviation = np.sqrt(
        np.dot(np.array(weights).T, np.dot(downside_returns.cov(), np.array(weights)))
    ) * np.sqrt(252)
    sortino = (portfolio_returns - riskfree) / downside_deviation
    col1, col2, col3 = st.columns([0.2, 1, 3])
    query = "Sortino Ratio"
    with col2:
        st.metric(query, value=round(sortino, 2))
    with col3:
        st.write(
            "The **Sortino Ratio** is a variation of Sharpe Ratio, that only accounts for downside risk"
        )
        st.write("A higher Sortino Ratio indicates better risk-adjusted performance.")
        st.write(
            "if Sortino > Sharpe, portfolio volatilty is more towards positive returns (Optimal Case)"
        )
        st.write(
            "if Sortino < Sharpe, portfolio has significant negative volatilty Really Bad)"
        )
with tab3:
    query = "Maximum Drawdown"
    col1, col2, col3 = st.columns([0.2, 1, 3])
    with col2:
        st.metric(
            query,
            value=str(round(max_drawdown * 100, 2)) + "%",
            delta=round(max_drawdown_value, 2),
        )
    with col3:
        st.write(
            "The **Maximum Drawdown (MDD)** measures the largest peak-to-trough decline in the value of a portfolio during a specific period."
        )
        st.write(
            "It helps assess the hypothetical risk of severe losses in a portfolio's value."
        )
        st.write("Lower the Maximum Drawdown, lower the risk exposure to losses")
        st.write(
            "A high MDD (25%+) indicates higher risk, portfolio has experienced significant volatility."
        )
        st.write(
            "A low MDD (10% to 15%) indicates lower risk, because of stable price changes."
        )

with tab4:
    confidence_level = 0.95
    z_score = norm.ppf(1 - confidence_level)
    var_relative = -(
        np.sum(returns * weights, axis=1).mean() * 10
        + z_score * portfolio_sd / np.sqrt(252) * np.sqrt(10)
    )
    var_absolute = sum(values) * var_relative

    col1, col2, col3 = st.columns([0.2, 1, 3])
    with col2:
        st.metric(
            "10-day Value at Risk (at .95)",
            value=str(round(var_relative * 100, 2)) + "%",
            delta=round(-var_absolute, 0),
        )
    with col3:
        st.write(
            "**Value at Risk (VaR)** estimates the potential maximum loss in a portfolio over a specified time period with a given confidence level."
        )
        st.write(
            "It is widely used to gauge the risk of loss on a portfolio's investments."
        )
        st.write(
            "A lower VaR indicates lower potential losses under normal market conditions (Indicates Lower Risk)."
        )
        st.write(
            "A higher VaR signals the portfolio is exposed to larger potential losses (Indicates Higher Risk)."
        )
        st.write(
            "Note that VaR does not account for extreme tail risks or unusual market conditions."
        )
