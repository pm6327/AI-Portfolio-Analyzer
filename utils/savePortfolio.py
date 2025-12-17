import json
import os
from utils.mongo_manager import MongoManager
import sqlite3
def save_portfolio(portfolio, stocks, weights_dict, std, max_drawdown):
    # Prepare dictionary for JSON serialization
    weights_dict = {}
    tickers = portfolio.Ticker
    values = portfolio.Value
    for ticker, value in zip(tickers, values):
        weights_dict[ticker] = value
        
    file_path = os.path.join('utils', 'portfolio_metrics.json')
    #  Create a dictionary to store metrics for all tickers
    portfolio_metrics = []

    # Iterate through each ticker to save its corresponding metrics
    for ticker in tickers:
        # Build the metrics dictionary dynamically for each ticker
        metrics = {ticker : {
            "Current Stock Price": round(stocks[ticker].tail(1).values[0], 2),  # Get the latest stock price
            "Portfolio Weight": round(weights_dict[ticker], 2),  # Get the weight of the ticker in the portfolio
            "Annualized Volatility": f"{std[ticker]}%",  # Annualized volatility
            "Maximum Drawdown": f"{round(max_drawdown[ticker] * 100, 2)}%"  # Maximum drawdown
        }}

        # Add the metrics for each ticker into the portfolio_metrics dictionary
        portfolio_metrics.append(metrics)
    
    with open(file_path, 'w') as file:
        json.dump(portfolio_metrics, file, indent=4)
        
    
    try:
        mongo_db = MongoManager(collection_name="portfolio_metrics",db_name="portfolio_data")
        ids = mongo_db.insert_documents(portfolio_metrics)
        mongo_db.close_connection
        return "Saved Successfully!"
    except Exception as e:
        print(f"An error Occured while inserting portfolio metrics: {e}")
        return "Error"