import streamlit as st
import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()
st.title("📰 Live Stock News")

st.caption("Real-time stock news with summaries")

# =============================
# CONFIG
# =============================

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# =============================
# User input
# =============================

symbol = st.text_input("Enter Stock / Company Name", "Tesla")

auto_refresh = st.toggle("Auto Refresh News", value=True)
refresh_rate = st.slider("Refresh every (seconds)", 30, 300, 60)

# =============================
# Fetch NewsAPI articles
# =============================


@st.cache_data(ttl=60)
def fetch_news(query):

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&"
        f"sortBy=publishedAt&"
        f"language=en&"
        f"pageSize=20&"
        f"apiKey={NEWS_API_KEY}"
    )

    response = requests.get(url)
    data = response.json()

    # debug remove later
    # st.write(data)

    return data.get("articles", [])


# =============================
# UI
# =============================

if symbol:

    news_articles = fetch_news(symbol)

    if not news_articles:
        st.warning("No news articles found.")
    else:

        st.subheader(f"Latest News for {symbol}")

        for article in news_articles[:20]:

            title = article.get("title")
            description = article.get("description")
            url = article.get("url")
            source = article.get("source", {}).get("name", "Unknown")

            if not title or not url:
                continue

            with st.container():

                st.markdown(f"### {title}")
                st.caption(f"Source: {source}")

                if description:
                    st.write(description)
                else:
                    st.write("Summary not available.")

                st.link_button("Read Full Article", url)

                st.divider()

    # =============================
    # Auto refresh
    # =============================

    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

#  Yaha se api li hai
# https://newsapi.org/account
