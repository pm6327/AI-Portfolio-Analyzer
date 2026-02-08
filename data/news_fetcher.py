import requests
import os
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")


def get_stock_news(symbol):
    """
    Fetch ONLY stock-market-related news for any given company/ticker.
    Works globally for US, India, EU stocks.
    """

    if not NEWS_API_KEY:
        return []

    # Force financial context dynamically
    financial_query = (
        f'("{symbol}" AND ("stock" OR "share price" OR "shares" OR '
        '"earnings" OR "revenue" OR "profit" OR "loss" OR '
        '"analyst" OR "price target" OR "market" OR '
        '"quarter results" OR "guidance" OR "IPO" OR "valuation"))'
    )

    url = (
        "https://newsapi.org/v2/everything?"
        f"q={financial_query}&"
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

        articles = data.get("articles", [])

        # Return only headlines (used by sentiment engine)
        headlines = [a.get("title", "") for a in articles if a.get("title")]

        return headlines

    except Exception as e:
        print("News fetch error:", e)
        return []
