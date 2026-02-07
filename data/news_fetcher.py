import requests
import os

from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")


def get_stock_news(symbol):

    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"

    try:
        res = requests.get(url)
        data = res.json()

        # handle API errors
        if "articles" not in data:
            return []

        headlines = [a.get("title", "") for a in data["articles"] if a.get("title")]

        return headlines

    except Exception as e:
        print("News fetch error:", e)
        return []


#
