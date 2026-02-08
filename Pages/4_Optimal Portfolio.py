import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


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
st.markdown(
    """
<div class="header-card">
<h2>🚀 Optimal Portfolio Allocation</h2>
<p>Monte Carlo simulation of 3000 portfolio weight combinations to identify the efficient frontier and best allocation.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ---------- SESSION DATA ----------
portfolio = st.session_state["portfolio"]
tickers = st.session_state["tickers"]
values = st.session_state["values"]
weights = st.session_state["weights"]
weights_dict = st.session_state["weights_dict"]
stocks = st.session_state["stocks"]
returns = st.session_state["returns"]


# ---------- GENERAL CALCULATIONS ----------
covariance = returns.cov()

portfolio_sd = np.sqrt(
    np.dot(np.array(weights).T, np.dot(covariance, np.array(weights)))
) * np.sqrt(252)

portfolio_returns = np.dot(weights, returns.mean() * 252)
riskfree = 0.023


# ---------- MONTE CARLO SIMULATION ----------
def cal_scenario():

    def hypothetical_portfolio(weights, returns, covariance):
        inner_dot = np.dot(covariance, weights)
        var = np.dot(weights.T, inner_dot)
        hypo_volatility = np.sqrt(var) * np.sqrt(252)
        hypo_returns = np.dot(weights, returns.mean() * 252)
        hypo_sharpe = (hypo_returns - riskfree) / hypo_volatility
        return hypo_returns, hypo_volatility, hypo_sharpe

    num_portfolios = 3000
    num_assets = len(returns.columns)

    hypo_returns = []
    hypo_volatility = []
    hypo_weights = []
    hypo_sharpe = []

    for _ in range(num_portfolios):
        temp_weights = np.random.rand(num_assets)
        temp_weights /= np.sum(temp_weights)

        hypo_weights.append(temp_weights)

        port_return, port_vol, port_sharpe = hypothetical_portfolio(
            temp_weights, returns, covariance
        )

        hypo_returns.append(port_return)
        hypo_volatility.append(port_vol)
        hypo_sharpe.append(port_sharpe)

    results = pd.DataFrame(
        {
            "returns": hypo_returns,
            "volatility": hypo_volatility,
            "sharpe ratio": hypo_sharpe,
        }
    )

    for i, ticker in enumerate(tickers):
        results[f"{ticker}_weight"] = [w[i] for w in hypo_weights]

    return results.T


results = cal_scenario()

min_var_port = results[results.loc["volatility"].idxmin()]
optimal_port = results[results.loc["sharpe ratio"].idxmax()]


# ---------- CAL LINE ----------
max_vol = results.loc["volatility"].max()
cal_x = np.linspace(0.0, max_vol, 10)

slope = (optimal_port["returns"] - riskfree) / optimal_port["volatility"]
cal_y = riskfree + slope * cal_x


# ---------- VISUAL LAYOUT ----------
left, right = st.columns([3, 1.2])


# ---------- SCATTER PLOT ----------
with left:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results.loc["volatility"].values,
            y=results.loc["returns"].values,
            mode="markers",
            marker=dict(size=6, opacity=0.7),
            name="Simulated Portfolios",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[min_var_port.T["volatility"]],
            y=min_var_port.T[["returns"]],
            mode="markers",
            marker=dict(size=12),
            name="Minimum Volatility",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[optimal_port.T["volatility"]],
            y=optimal_port.T[["returns"]],
            mode="markers",
            marker=dict(size=12),
            name="Max Sharpe Portfolio",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=cal_x,
            y=cal_y,
            mode="lines",
            name="Capital Allocation Line",
        )
    )

    def calculate_cal_portfolios(
        optimal_port, risk_free_rate, risk_aversion=3, num_points=100
    ):
        opt_return = optimal_port["returns"]
        opt_vol = optimal_port["volatility"]

        weights = np.linspace(0, 2, num_points)

        cal_port = pd.DataFrame(
            {
                "weight": weights,
                "rf_weight": 1 - weights,
                "volatility": weights * opt_vol,
                "returns": risk_free_rate + weights * (opt_return - risk_free_rate),
            }
        )

        cal_port["utility"] = cal_port["returns"] - 0.5 * risk_aversion * (
            cal_port["volatility"] ** 2
        )

        return cal_port.T

    cal_port = calculate_cal_portfolios(optimal_port, riskfree)
    investors_port = cal_port[cal_port.loc["utility"].idxmax()]

    fig.add_trace(
        go.Scatter(
            x=[investors_port.T["volatility"]],
            y=investors_port.T[["returns"]],
            mode="markers",
            marker=dict(size=14, symbol="star"),
            name="Investor Optimal Portfolio",
        )
    )

    fig.update_layout(
        xaxis_title="Volatility",
        yaxis_title="Expected Returns",
        template="plotly_dark",
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------- SIDE INSIGHTS ----------
with right:

    st.markdown("### 📌 Portfolio Insights")

    st.info("Red → lowest volatility portfolio")
    st.info("Orange → highest Sharpe ratio")
    st.info("Star → investor optimal mix")

    st.markdown("### 📊 Current Portfolio")

    st.metric("Return", f"{round(portfolio_returns*100,2)}%")
    st.metric("Risk", f"{round(portfolio_sd*100,2)}%")


# ---------- BREAKDOWN TABS ----------
st.markdown("### 🧮 Allocation Breakdown")

tabs = st.tabs(["Investor Optimal", "Max Return Portfolio", "Minimum Risk Portfolio"])


# ----- INVESTOR PORT
with tabs[0]:
    port1 = [investors_port[0] * x for x in weights]
    s = (investors_port[3] - riskfree) / investors_port[2]

    c1, c2, c3 = st.columns(3)
    c1.metric("Returns", f"{round(investors_port[3]*100,2)}%")
    c2.metric("Risk", f"{round(investors_port[2]*100,2)}%")
    c3.metric("Sharpe", round(s, 2))

    cols = st.columns(len(port1))

    for i, weight in enumerate(port1):
        with cols[i]:
            st.metric(
                tickers[i],
                f"$ {round(weight*sum(values),0)}",
                delta=round(weight * sum(values) - values[i], 0),
            )


# ----- MAX RETURN
with tabs[1]:
    port2 = list(optimal_port[3:])

    c1, c2, c3 = st.columns(3)
    c1.metric("Returns", f"{round(optimal_port[0]*100,2)}%")
    c2.metric("Risk", f"{round(optimal_port[1]*100,2)}%")
    c3.metric("Sharpe", round(optimal_port[2], 2))

    cols = st.columns(len(port2))

    for i, weight in enumerate(port2):
        with cols[i]:
            st.metric(
                tickers[i],
                f"$ {round(weight*sum(values),0)}",
                delta=round(weight * sum(values) - values[i], 0),
            )


# ----- MIN RISK
with tabs[2]:
    port3 = list(min_var_port[3:])

    c1, c2, c3 = st.columns(3)
    c1.metric("Returns", f"{round(min_var_port[0]*100,2)}%")
    c2.metric("Risk", f"{round(min_var_port[1]*100,2)}%")
    c3.metric("Sharpe", round(min_var_port[2], 2))

    cols = st.columns(len(port3))

    for i, weight in enumerate(port3):
        with cols[i]:
            st.metric(
                tickers[i],
                f"$ {round(weight*sum(values),0)}",
                delta=round(weight * sum(values) - values[i], 0),
            )
