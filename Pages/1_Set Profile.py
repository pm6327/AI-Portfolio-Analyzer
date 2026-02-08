import streamlit as st
import pandas as pd
import yfinance as yf


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
.input-card {
    padding: 16px;
    border-radius: 12px;
    background-color: #020617;
    border: 1px solid #1e293b;
    margin-bottom: 12px;
}
.section-title {
    font-size: 20px;
    font-weight: 600;
    margin-top: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------- HEADER ----------
st.markdown(
    """
<div class="header-card">
<h2>📊 Stock Portfolio Risk Evaluation</h2>
<p>Build your portfolio and evaluate its risk exposure, volatility, and performance.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("### 🧾 Enter Stock tickers and their portfolio value")


# ---------- SESSION INIT ----------
if "portfolio" not in st.session_state:
    st.session_state.portfolio = pd.DataFrame({"Ticker": [""], "Value": [0.0]})

if "file" not in st.session_state:
    st.session_state.file = pd.DataFrame({"Ticker": [""]})

if "disable" not in st.session_state:
    st.session_state.disable = True


# ---------- FUNCTIONS ----------
def add_stock():
    new_row = pd.DataFrame({"Ticker": [""], "Value": [0.0]})
    st.session_state.portfolio = pd.concat(
        [st.session_state.portfolio, new_row], ignore_index=True
    )


def remove_stock(index):
    st.session_state.portfolio = st.session_state.portfolio.drop(index).reset_index(
        drop=True
    )


def is_valid(ticker, value):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if "symbol" in info and info["symbol"] == ticker and value > 0:
            return True
        else:
            return False
    except Exception:
        return False


# ---------- PORTFOLIO INPUT UI ----------
st.markdown("### 📥 Portfolio Builder")

for index, _ in st.session_state.portfolio.iterrows():

    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2, 1, 0.5], vertical_alignment="bottom")

        with col1:
            ticker = st.text_input(
                "Ticker",
                value=st.session_state.portfolio.at[index, "Ticker"],
                key=f"ticker_{index}",
                placeholder="e.g. RELIANCE / AAPL",
            )
            st.session_state.portfolio.at[index, "Ticker"] = ticker.upper()

        with col2:
            value = st.number_input(
                "Value Invested",
                min_value=0.0,
                value=float(st.session_state.portfolio.at[index, "Value"]),
                format="%.2f",
                key=f"value_{index}",
            )
            st.session_state.portfolio.at[index, "Value"] = value

        with col3:
            if index > 0:
                if st.button("🗑️", key=f"remove_{index}"):
                    remove_stock(index)
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


st.session_state.portfolio = pd.DataFrame(st.session_state.portfolio)


# ---------- ACTION BAR ----------
st.markdown("### ⚙️ Portfolio Actions")

a1, a2, a3, a4 = st.columns([1.5, 1.5, 2, 2])

with a1:
    st.button("➕ Add Stock", on_click=add_stock)

with a2:
    if st.button("🔍 Validate Portfolio"):

        p = st.session_state.portfolio
        ticker_input = p["Ticker"]
        values = p["Value"]

        if ticker_input.empty:
            st.error("Add at least one stock.")
        else:
            valid_count = 0
            for index, row in p.iterrows():
                ticker = row["Ticker"]
                value = row["Value"]

                if is_valid(ticker, value):
                    valid_count += 1
                else:
                    st.error(f"{ticker} invalid — check spelling or value")

            if valid_count == len(p):
                st.success("Portfolio validated successfully.")
                st.session_state.disable = False

with a3:
    st.page_link(
        "Pages/2_Stock Charts.py",
        label="📈 Go to Stock Charts",
        disabled=st.session_state.disable,
    )

with a4:
    if st.session_state.disable:
        st.info("Validate portfolio to unlock analysis")


# ---------- SUMMARY PANEL ----------
if not st.session_state.disable:

    st.markdown("### 📊 Portfolio Summary")

    total_value = st.session_state.portfolio["Value"].sum()
    total_assets = len(st.session_state.portfolio)

    s1, s2 = st.columns(2)

    s1.metric("Total Investment", f"₹ {round(total_value,2)}")
    s2.metric("Assets Added", total_assets)
