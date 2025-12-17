import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


#To prevent access from sidebar before inputting values
if st.session_state.disable:
    st.info('Please enter your input on the Home page and try again.')
    st.stop()

st.title("Optimal Portfolio Allocation")

st.write("This is a scatterplot of 3000 scenarios with randomised weights of each asset to determine the optimal portfolio distribution")
# ----- LOCAL VARIABLES -----
portfolio = st.session_state['portfolio']
tickers = st.session_state['tickers']
values = st.session_state['values']
weights = st.session_state['weights']
weights_dict = st.session_state['weights_dict']
stocks = st.session_state['stocks']
returns = st.session_state['returns']


# ----- GENERAL CALCULATIONS -----
covariance = returns.cov()
portfolio_sd=np.sqrt(np.dot(np.array(weights).T, np.dot(covariance, np.array(weights)))) * np.sqrt(252)
portfolio_returns = np.dot(weights, returns.mean()*252) 
riskfree = 0.023

def cal_scenario():
      def hypothetical_portfolio(weights, returns, covariance):
            inner_dot = np.dot(covariance, weights)
            var = np.dot(weights.T, inner_dot)
            hypo_volatility = np.sqrt(var)* np.sqrt(252)
            hypo_returns = np.dot(weights, returns.mean()*252)
            hypo_sharpe = (hypo_returns-riskfree)/hypo_volatility
            return hypo_returns, hypo_volatility, hypo_sharpe

      # Set up parameters
      num_portfolios = 3000 # For simplicity, showing a smaller number of portfolios
      num_assets = len(returns.columns) 
      hypo_returns = []
      hypo_volatility = []
      hypo_weights = []
      hypo_sharpe = []

      for _ in range(num_portfolios):
            temp_weights = np.random.rand(num_assets)
            temp_weights /= np.sum(temp_weights)  # Normalize weights to sum to 1
            hypo_weights.append(temp_weights)
            
            # Calculate return and volatility
            port_return, port_vol, port_sharpe = hypothetical_portfolio(temp_weights, returns, covariance)
            hypo_returns.append(port_return)
            hypo_volatility.append(port_vol)
            hypo_sharpe.append(port_sharpe)

      # Create a DataFrame for results
      results = pd.DataFrame({
      'returns': hypo_returns,
      'volatility': hypo_volatility,
      'sharpe ratio': hypo_sharpe
      })

      # Add weights to the DataFrame
      for i, ticker in enumerate(tickers):
            results[f'{ticker}_weight'] = [w[i] for w in hypo_weights]

      return results.T

results = cal_scenario()
min_var_port = results[results.loc['volatility'].idxmin()]
optimal_port = results[results.loc['sharpe ratio'].idxmax()]

max_vol = results.loc["volatility"].max()  # Extend beyond max volatility
cal_x = np.linspace(0., max_vol, 10)
slope = (optimal_port['returns'] - riskfree) / optimal_port['volatility']
cal_y = riskfree + slope * cal_x

c1, c2 = st.columns([3,1], vertical_alignment="center")

with c1:
    cal = go.Figure()
    cal.add_trace(go.Scatter(x=results.loc["volatility"].values, y=results.loc["returns"].values, mode='markers',marker=dict(color='lightgrey', size=6, opacity=0.8), name='Portfolios'))
    cal.add_trace(go.Scatter(x=[min_var_port.T["volatility"]],y=min_var_port.T[["returns"]], mode = 'markers', marker=dict(color='crimson', size=12, opacity =1), name = "Minimum Volatility Portfolio"))
    cal.add_trace(go.Scatter(x=[optimal_port.T["volatility"]],y=optimal_port.T[["returns"]], mode = 'markers', marker=dict(color='orange', size=12, opacity=1), name = "Optimal Portfolio"))
    cal.add_trace(go.Scatter(x=cal_x, y=cal_y, mode='lines', line=dict(color='purple', width=2.5), name='Capital Allocation Line'))
    cal.update_layout(xaxis_title="Volatility",
                    yaxis_title="Returns",
                    template = 'plotly_dark',
                    hovermode='closest',
                    showlegend=True)

    def calculate_cal_portfolios(optimal_port, risk_free_rate, risk_aversion=3, num_points=100):
        # Extract optimal portfolio characteristics
        opt_return = optimal_port['returns']
        opt_vol = optimal_port['volatility']
        
        # Create weights from 0 to 2 (0% to 200% in the risky portfolio)
        weights = np.linspace(0, 2, num_points)
        
        # Calculate CAL portfolios
        cal_port = pd.DataFrame({
                'weight': weights,
                'rf_weight': 1 - weights,
                'volatility': weights * opt_vol,
                'returns': risk_free_rate + weights * (opt_return - risk_free_rate)
        })
        
        # Calculate utility for each portfolio
        cal_port['utility'] = cal_port['returns'] - 0.5 * risk_aversion * (cal_port['volatility'] ** 2)
        cal_port = cal_port.T
        return cal_port

    cal_port = calculate_cal_portfolios(optimal_port, riskfree)
    investors_port = cal_port[cal_port.loc['utility'].idxmax()]
    cal.add_trace(go.Scatter(x=[investors_port.T["volatility"]],y=investors_port.T[["returns"]], mode = 'markers', marker=dict(color='green', size=12, opacity=1, symbol='star'), name = "Investors Portfolio"))
    st.plotly_chart(cal)

with c2:
    st.write("The :red[crimson] point is the safest portfolio (Least Volatile)")
    st.write("The :orange[orange] point is the portfolio with the best risk-adjusted return")
    st.write("The small :green[green] star is the portfolio with the most optimal risk-return tradeoff")

portfolio_list = ["Optimal Portfolio", "Highest-Return Portfolio", "Lowest-Risk Portfolio"]
tab1, tab2, tab3 = st.tabs(portfolio_list)

with tab1:
    port1 = [investors_port[0] * x for x in weights]
    s = (investors_port[3]-riskfree)/investors_port[2]
    col1, col2, col3 = st.columns(3)
    col1.write("***Returns:*** " +str(round(investors_port[3]*100, 2))+"%")
    col2.write("***Risk:*** " +str(round(investors_port[2]*100, 2))+"%")
    col3.write("***Sharpe Ratio:*** " +str(round(s, 2)))

    if investors_port[1]*100 > 0:
        st.write(str(round(investors_port[1]*100,2))+ "% of the portfolio, which is $"+str(round(investors_port[1]*sum(values),0))+" needs to be allocated to a risk free asset, like a Treasury bill to lower the volatility of the portfolio")
    else:
        diff = round((investors_port[0] - 1)*sum(values), 0)
        st.write("$"+str(diff)+" worth of additional investment will help to lower portfolio risk ")
        #diff = port1['values'].sum() - sum(values)

    cols = st.columns(len(port1))

    for i,weight in enumerate(port1):
        with cols[i]:
            st.metric(label=tickers[i], value = "$"+ str(round(weight*sum(values), 0)), delta = round(weight*sum(values) - values[i],0))
with tab2:
    port2 = list(optimal_port[3:])
    col1, col2, col3 = st.columns(3)
    col1.write("***Returns:*** " +str(round(optimal_port[0]*100, 2))+"%")
    col2.write("***Risk:*** " +str(round(optimal_port[1]*100, 2))+"%")
    col3.write("***Sharpe Ratio:*** " +str(round(optimal_port[2], 2)))

    cols = st.columns(len(port2))
    for i,weight in enumerate(port2):
        with cols[i]:
            st.metric(label=tickers[i], value = "$"+ str(round(weight*sum(values), 0)), delta = round(weight*sum(values) - values[i],0))
with tab3:
    port3 = list(min_var_port[3:])
    col1, col2, col3 = st.columns(3)
    col1.write("***Returns:*** " +str(round(min_var_port[0]*100, 2))+"%")
    col2.write("***Risk:*** " +str(round(min_var_port[1]*100, 2))+"%")
    col3.write("***Sharpe Ratio:*** " +str(round(min_var_port[2], 2)))

    cols = st.columns(len(port2))

    for i,weight in enumerate(port3):
        with cols[i]:
            st.metric(label=tickers[i], value = "$"+ str(round(weight*sum(values), 0)), delta = round(weight*sum(values) - values[i],0))

