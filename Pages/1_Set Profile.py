import streamlit as st
import pandas as pd
import yfinance as yf


# ---------- Page Config ----------
st.set_page_config(
    page_title="Stock Portfolio Risk Evaluation", page_icon="📊", layout="wide"
)

# ---------- Custom Styling ----------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .section-text {
            font-size: 15px;
            color: #6b7280;
            margin-bottom: 20px;
        }
        .stButton button {
            border-radius: 8px;
            font-weight: 600;
            padding: 8px 16px;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown(
    '<div class="main-title">📊 Stock Portfolio Risk Evaluation</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-text">Enter stock tickers and their corresponding investment value to analyze your portfolio risk.</div>',
    unsafe_allow_html=True,
)

# ---------- Session State ----------
if "portfolio" not in st.session_state:
    st.session_state.portfolio = pd.DataFrame({"Ticker": [""], "Value (INR)": [0.0]})

if "disable_nav" not in st.session_state:
    st.session_state.disable_nav = True


# ---------- Helper Functions ----------
def add_stock():
    new_row = pd.DataFrame({"Ticker": [""], "Value (INR)": [0.0]})
    st.session_state.portfolio = pd.concat(
        [st.session_state.portfolio, new_row], ignore_index=True
    )


def remove_stock(index):
    st.session_state.portfolio = st.session_state.portfolio.drop(index).reset_index(
        drop=True
    )


def is_valid(ticker, value):
    try:
        ticker = ticker.strip().upper()
        if ticker == "" or value <= 0:
            return False

        stock = yf.Ticker(ticker)
        info = stock.fast_info  # faster & more stable than .info

        return info is not None
    except:
        return False


# ---------- Portfolio Input Form ----------
with st.container(border=True):
    st.subheader("Portfolio Inputs")

    for index, _ in st.session_state.portfolio.iterrows():
        col1, col2, col3 = st.columns([3, 2, 0.7])

        with col1:
            ticker = st.text_input(
                "Stock Ticker",
                value=st.session_state.portfolio.at[index, "Ticker"],
                key=f"ticker_{index}",
                placeholder="e.g., AAPL, TCS, INFY",
            )
            st.session_state.portfolio.at[index, "Ticker"] = ticker

        with col2:
            value = st.number_input(
                "Investment Value (₹)",
                min_value=0.0,
                value=float(st.session_state.portfolio.at[index, "Value (INR)"]),
                format="%.2f",
                key=f"value_{index}",
            )
            st.session_state.portfolio.at[index, "Value (INR)"] = value

        with col3:
            st.write("")  # spacing
            if index > 0:
                if st.button("Remove", key=f"remove_{index}"):
                    remove_stock(index)
                    st.rerun()


# ---------- Action Buttons ----------
st.divider()

col1, col2, col3 = st.columns([1.2, 1.2, 2])

with col1:
    st.button("➕ Add Stock", on_click=add_stock, use_container_width=True)

with col2:
    if st.button("✔ Validate Portfolio", use_container_width=True):
        portfolio = st.session_state.portfolio
        valid_all = True

        for _, row in portfolio.iterrows():
            ticker = row["Ticker"]
            value = row["Value (INR)"]

            if is_valid(ticker, value):
                st.success(f"{ticker.upper()} validated successfully")
            else:
                st.error(f"{ticker} is invalid or value is incorrect")
                valid_all = False

        st.session_state.disable_nav = not valid_all

with col3:
    st.page_link(
        "Pages/2_Stock Charts.py",
        label="📈 Continue to Stock Charts",
        disabled=st.session_state.disable_nav,
    )
