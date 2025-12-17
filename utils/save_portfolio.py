import json
import os
import numpy as np
from scipy.stats import norm
from utils.mongo_manager import MongoManager
import sqlite3

def save_risk_metrics(portfolio, stocks, weights_dict, returns, riskfree, values, max_drawdown, portfolio_returns, portfolio_sd, tickers):
    # Convert weights_dict to numpy array for matrix operations
    
    weights = np.array([weights_dict[ticker] for ticker in tickers])
    
    # Prepare dictionary for saving
    risk_metrics = {}

    # ----- Sharpe Ratio -----
    sharpe = (portfolio_returns - riskfree) / portfolio_sd
    risk_metrics["Sharpe Ratio"] = round(sharpe, 2)

    # ----- Sortino Ratio -----
    mar = 0  # Minimum Acceptable Return for Sortino
    downside_returns = returns.where(returns < 0)
    downside_deviation = np.sqrt(np.dot(weights.T, np.dot(downside_returns.cov(), weights))) * np.sqrt(252)
    sortino = (portfolio_returns - riskfree) / downside_deviation
    risk_metrics["Sortino Ratio"] = round(sortino, 2)

    # ----- Maximum Drawdown -----
    max_drawdown_value = max_drawdown * sum(values)
    risk_metrics["Maximum Drawdown"] = {
        "Percentage": f"{round(max_drawdown * 100, 2)}%",
        "Delta": round(max_drawdown_value, 2)
    }

    # ----- Value-at-Risk (VaR) -----
    confidence_level = 0.95
    z_score = norm.ppf(1 - confidence_level)
    var_relative = -(np.sum(returns * weights, axis=1).mean() * 10 + z_score * portfolio_sd / np.sqrt(252) * np.sqrt(10))
    var_absolute = sum(values) * var_relative
    risk_metrics["10-day Value at Risk (at .95)"] = {
        "Percentage": f"{round(var_relative * 100, 2)}%",
        "Delta": round(-var_absolute, 0)
    }
    file_path = os.path.join('utils', 'risk_metrics.json')

    with open(file_path, 'w') as file:
        json.dump(risk_metrics, file, indent=4)
    try:
        mongo_db = MongoManager(collection_name="risk_metrics",db_name="portfolio_data")
        ids = mongo_db.insert_document(risk_metrics)
        mongo_db.close_connection
        
        return "Successfully saved risk_metric"
    except Exception as e:
        print(f"There was some error :{e}")
        raise e