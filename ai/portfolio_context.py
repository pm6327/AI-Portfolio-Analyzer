def build_portfolio_context(portfolio, predictions, risk_metrics):
    context = f"""
    Portfolio Summary:
    {portfolio}

    Model Predictions:
    {predictions}

    Risk Metrics:
    {risk_metrics}
    """

    return context
