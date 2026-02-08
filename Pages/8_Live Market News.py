import streamlit as st
import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()

# ---------- PAGE CONFIG ----------
st.set_page_config(layout="wide")

st.markdown(
    """
<style>
.header-card {
    padding: 18px;
    border-radius: 14px;
    background: linear-gradient(135deg,#020617,#0f172a);
    border: 1px solid #1e293b;
}
.news-card {
    padding: 16px;
    border-radius: 12px;
    background-color: #020617;
    border: 1px solid #1e293b;
    margin-bottom: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------- HEADER ----------
st.markdown(
    """
<div class="header-card">
<h2>📰 Live Stock Market News</h2>
<p>Real-time financial & stock-market-relevant news for any company or ticker.</p>
</div>
""",
    unsafe_allow_html=True,
)


# =============================
# CONFIG
# =============================
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not NEWS_API_KEY:
    st.error("NEWS_API_KEY missing in .env file")
    st.stop()


# =============================
# INPUT
# =============================
st.markdown("### 🔎 Search Stock News")

col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    symbol = st.text_input(
        "Enter Stock / Company Name", placeholder="AAPL, Tesla, Reliance"
    )

with col2:
    auto_refresh = st.toggle("Auto Refresh", value=False)

with col3:
    refresh_rate = st.slider("Seconds", 30, 300, 60)


# =============================
# STOCK NEWS FETCHER
# =============================
@st.cache_data(ttl=60)
def fetch_stock_news(query):

    financial_context = (
        f'("{query}" AND ("stock" OR "share price" OR "shares" OR '
        '"earnings" OR "revenue" OR "profit" OR "loss" OR '
        '"analyst" OR "price target" OR "market" OR '
        '"quarter results" OR "guidance" OR "valuation"))'
    )

    url = (
        "https://newsapi.org/v2/everything?"
        f"q={financial_context}&"
        "sortBy=publishedAt&"
        "language=en&"
        "pageSize=25&"
        f"apiKey={NEWS_API_KEY}"
    )

    try:
        res = requests.get(url)
        data = res.json()

        if res.status_code != 200:
            return []

        return data.get("articles", [])

    except:
        return []


# =============================
# UI DISPLAY
# =============================
if symbol:

    with st.spinner("Fetching financial news..."):
        news_articles = fetch_stock_news(symbol)

    if not news_articles:
        st.warning("No stock-related articles found.")
    else:

        st.markdown(f"### 📈 Latest Stock News — {symbol.upper()}")

        for article in news_articles:

            title = article.get("title")
            description = article.get("description")
            url = article.get("url")
            source = article.get("source", {}).get("name", "Unknown")
            published = article.get("publishedAt", "")[:10]

            if not title or not url:
                continue

            with st.container():
                st.markdown('<div class="news-card">', unsafe_allow_html=True)

                st.markdown(f"#### {title}")
                st.caption(f"{source} • {published}")

                if description:
                    st.write(description)
                else:
                    st.write("No financial summary available.")

                st.link_button("Read Full Article", url)

                st.markdown("</div>", unsafe_allow_html=True)


# =============================
# AUTO REFRESH
# =============================
if symbol and auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
