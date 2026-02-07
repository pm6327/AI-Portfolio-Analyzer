from transformers import pipeline

# load once
sentiment_model = pipeline("sentiment-analysis")


def analyze_sentiment(news):
    scores = []
    for n in news:
        result = sentiment_model(n)[0]
        scores.append(
            result["score"] if result["label"] == "POSITIVE" else -result["score"]
        )
    return sum(scores) / len(scores) if scores else 0
