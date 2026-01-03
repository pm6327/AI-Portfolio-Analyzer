import json
import os
import numpy as np
from scipy.stats import norm
from utils.mongo_manager import MongoManager


def save_risk_metrics(
    portfolio,
    stocks,
    weights_dict,
    returns,
    riskfree,
    values,
    max_drawdown,
    portfolio_returns,
    portfolio_sd,
    tickers,
):
    # ---------- VALIDATION ----------
    if returns is None or returns.empty:
        raise ValueError("Returns data is empty")

    valid_tickers = [t for t in tickers if t in returns.columns]
    if not valid_tickers:
        raise ValueError("No valid tickers found in returns data")

    returns = returns[valid_tickers]
    weights = np.array([weights_dict.get(t, 0) for t in valid_tickers])

    if portfolio_sd is None or np.isnan(portfolio_sd) or portfolio_sd <= 0:
        raise ValueError("Invalid portfolio standard deviation")

    total_value = sum(values)

    # ---------- METRICS ----------
    risk_metrics = {}

    sharpe = (portfolio_returns - riskfree) / portfolio_sd
    risk_metrics["Sharpe Ratio"] = round(float(sharpe), 2)

    downside_returns = returns.clip(upper=0)
    downside_cov = downside_returns.cov()

    if downside_cov.empty:
        risk_metrics["Sortino Ratio"] = "N/A"
    else:
        downside_dev = np.sqrt(
            np.dot(weights.T, np.dot(downside_cov, weights))
        ) * np.sqrt(252)
        risk_metrics["Sortino Ratio"] = (
            round((portfolio_returns - riskfree) / downside_dev, 2)
            if downside_dev > 0
            else "N/A"
        )

    risk_metrics["Maximum Drawdown"] = {
        "Percentage": f"{round(max_drawdown * 100, 2)}%",
        "Delta": round(max_drawdown * total_value, 2),
    }

    confidence_level = 0.95
    z_score = norm.ppf(1 - confidence_level)

    portfolio_daily_returns = np.sum(returns * weights, axis=1)
    mean_return = portfolio_daily_returns.mean()

    var_relative = -(
        mean_return * 10 + z_score * portfolio_sd / np.sqrt(252) * np.sqrt(10)
    )

    risk_metrics["10-day Value at Risk (95%)"] = {
        "Percentage": f"{round(var_relative * 100, 2)}%",
        "Delta": round(-total_value * var_relative, 0),
    }

    # ---------- SAVE TO FILE ----------
    file_path = os.path.join("utils", "risk_metrics.json")
    with open(file_path, "w") as f:
        json.dump(risk_metrics, f, indent=4)

    # ---------- SAVE TO MONGODB (OPTIONAL) ----------
    try:
        mongo_db = MongoManager(
            collection_name="risk_metrics", db_name="portfolio_data"
        )
        mongo_db.insert_document(risk_metrics)
        mongo_db.close_connection()
    except Exception:
        # Do NOT crash Streamlit if MongoDB is down
        print("MongoDB unavailable. Skipping DB save.")

    return "Successfully saved risk metrics"
