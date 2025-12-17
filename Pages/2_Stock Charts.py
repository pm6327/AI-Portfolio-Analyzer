import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date
import plotly.graph_objects as go
from utils.savePortfolio import save_portfolio



#To prevent access from sidebar before inputting values
if st.session_state.disable:
    st.info('Please enter your input on the Home page and try again.')
    st.stop()

# ----- SETTING UP LOCAL VARIABLES -----
portfolio = st.session_state.portfolio
weights_dict = {}
if portfolio is not None:
    # Update weights_dict based on portfolio values
    tickers = portfolio.Ticker
    values = portfolio.Value

    for ticker in tickers:
        weight = portfolio.loc[portfolio["Ticker"] == ticker, 'Value'].values / sum(values)
        weights_dict[ticker] = float(weight)

    # Update portfolio DataFrame with weights
    portfolio['Weight'] = portfolio['Ticker'].map(weights_dict)
    st.session_state.portfolio = portfolio

tickers = portfolio.Ticker
values = portfolio.Value

for ticker in tickers:
    weight = portfolio.loc[portfolio["Ticker"] == ticker, 'Value'].values/sum(values)
    weights_dict[ticker] = float(weight)

portfolio['Weight'] = portfolio['Ticker'].map(weights_dict)
weights = list(weights_dict.values())
st.session_state.portfolio = portfolio

stocks = yf.download(tickers=list(tickers), period='5y')
stocks=stocks['Adj Close']
today = stocks.index[-1]
yday = stocks.index[-2]
returns= stocks.pct_change().dropna()
std = np.std(returns) * 100 * np.sqrt(252)  # Annualized volatility
cumulative_returns = (1 + returns).cumprod()
drawdowns = cumulative_returns / cumulative_returns.expanding().max() - 1
max_drawdown = drawdowns.min()

if "stocks" not in st.session_state:
    st.session_state["stocks"] = []

if "returns" not in st.session_state:
    st.session_state["returns"] = []

    
st.session_state['stocks'] = stocks
st.session_state['returns'] = returns
st.session_state['tickers'] = tickers
st.session_state['values'] = values
st.session_state['weights'] = weights
st.session_state['weights_dict'] = weights_dict

c1, c2, c3= st.columns([2,2,1], vertical_alignment="bottom")

 # Save the first ticker as an example
#----- Actual Page Code -----
with c1:
    frequency = st.selectbox(
        "Frequency of return",
        ("Daily", "Weekly", "Monthly", "Yearly")
    )
with c2:
    t = st.selectbox(
        "Select ticker",
        tickers
    )
with c3:
    st.page_link("Pages/3_Risk Analysis.py", label="Go to Risk Analysis →")

c1, c2 = st.columns([3,1], vertical_alignment="center")
with c1:
    st.title(t+" Summary")


# ----- FREQUENCY SET UP
f = stocks.pct_change().dropna()
daily_std = round(np.std(stocks[t].dropna()))
std = round(np.std(returns)*100*np.sqrt(252), 2)
risk = std.to_frame()
risk = risk.rename(columns = {risk.columns[0]:'Volatility'})
st.session_state['risk'] = risk

if frequency == "Daily":
    f = stocks.pct_change().dropna()
elif frequency == "Weekly":
    f = stocks.resample('W-FRI').last().pct_change().dropna()
elif frequency == "Monthly":
    f = stocks.resample('M').last().pct_change().dropna()
elif frequency == "Yearly":
    f = stocks.resample('Y').last().pct_change().dropna()

cumulative_returns = (1 + returns).cumprod()
rolling_max = cumulative_returns.expanding().max()
drawdowns = cumulative_returns / rolling_max - 1
max_drawdown = drawdowns.min()

# ----- STOCK CHARTS ------
col1, col2=st.columns([3,2])
with col1:
    tab1, tab2 = st.tabs(["Historical Stock Price", "Live Price"])
    with tab1:
        st.write( "Stock Price (Last 5 years)") 
        st.line_chart(stocks[t], height=350)
    with tab2:
        today = date.today()
        ohlc = yf.Ticker(t).history(start=yday, end=today, interval="2m")
        candlestick = go.Candlestick(x=ohlc.index,
                                 open=ohlc.Open,
                                 high=ohlc.High,
                                 low=ohlc.Low,
                                 close=ohlc.Close)

        layout = go.Layout(xaxis=dict(title='Date'),
                        yaxis=dict(title='Price'),
                        height=400)

        ohlc_chart = go.Figure(data=[candlestick], layout=layout)
        st.plotly_chart(ohlc_chart)

#----- NEWS -----
with col2:
    st.subheader(t+" News")
    topnews = yf.Ticker(t).news[0:3]
    for i, article in enumerate(topnews):
        title = article['title']
        link = article['link']
        markdown_news = f"[{title}]({link})"
        st.markdown("  "+markdown_news)
     
    st.write("") #For padding

    pct = 100*f[t].tail(1).values[0]
    
    # ----- METRICS -----
    subcol1, subcol2, subcol3=st.columns([0.2,1,1])
    with subcol2:
        st.metric("Current Stock Price", value = round(stocks[t].tail(1).values[0], 2), delta= str(round(pct, 2))+ "%")
        st.metric("Portfolio Weight", value = round(weights_dict[t], 2))
    with subcol3:
        st.metric("Annualised Volatilty", value = str(std[t])+"%")
        st.write(" ") #For padding
        st.metric("Maximum Drawdown", value = str(round(max_drawdown[t]*100, 2))+"%")

st.write(frequency +" returns (Last 5 years)")
st.line_chart(f[t], height = 140)

save_portfolio(portfolio, stocks, weights_dict, std, max_drawdown) 