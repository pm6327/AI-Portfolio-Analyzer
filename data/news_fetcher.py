import requests

API_KEY = "pub_fab80513fb25441aa5cb41efd039ed05"  # replace


def get_stock_news(symbol):

    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={API_KEY}"

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


# pub_fab80513fb25441aa5cb41efd039ed05
