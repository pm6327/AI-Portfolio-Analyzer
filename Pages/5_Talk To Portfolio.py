import streamlit as st
from utils.chatbot import stream_chat_response
import os
import json

# =============================
# Paths
# =============================

risk_metrics_path = os.path.join("utils", "risk_metrics.json")
portfolio_metrics_path = os.path.join("utils", "portfolio_metrics.json")
prediction_path = os.path.join("utils", "model_predictions.json")  # create later


# =============================
# Context Builder
# =============================


def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def get_user_context():

    risk_metrics = load_json(risk_metrics_path)
    portfolio_metrics = load_json(portfolio_metrics_path)
    predictions = load_json(prediction_path)

    context = {
        "risk_metrics": risk_metrics,
        "portfolio_metrics": portfolio_metrics,
        "model_predictions": predictions,
    }

    return context


# =============================
# Intent Detection (Agent Brain)
# =============================


def detect_intent(query):

    query = query.lower()

    intent_map = {
        "sell_recommendation": ["sell", "exit", "remove"],
        "risk_analysis": ["risk", "safe", "danger", "volatile"],
        "rebalance": ["rebalance", "adjust portfolio"],
        "future_outlook": ["future", "predict", "forecast"],
        "best_stock": ["best stock", "opportunity", "growth"],
    }

    for intent, keywords in intent_map.items():
        if any(word in query for word in keywords):
            return intent

    return "general"


# =============================
# Agent Reasoning Layer
# =============================


def generate_agent_prompt(user_query, context):

    intent = detect_intent(user_query)

    reasoning_rules = {
        "sell_recommendation": "Identify underperforming stocks and suggest exits.",
        "risk_analysis": "Explain risk exposure and potential drawdowns.",
        "rebalance": "Suggest asset allocation improvements.",
        "future_outlook": "Use model predictions to forecast portfolio direction.",
        "best_stock": "Identify strongest growth opportunities.",
        "general": "Answer using financial reasoning.",
    }

    system_prompt = f"""
    You are an AI Portfolio Copilot.

    Portfolio Data:
    {context["portfolio_metrics"]}

    Risk Data:
    {context["risk_metrics"]}

    Model Predictions:
    {context["model_predictions"]}

    Task:
    {reasoning_rules[intent]}

    Respond like a financial analyst.
    Give insights, not generic text.
    """

    return system_prompt


def generated_response(prompt):

    context = get_user_context()
    system_prompt = generate_agent_prompt(prompt, context)

    response = stream_chat_response(user_query=prompt, context=[system_prompt])

    return response


# =============================
# Streamlit UI
# =============================


def main():

    st.markdown(
        "<h1 style='background: linear-gradient(90deg,#0ea5e9,#6366f1); -webkit-background-clip: text; color: transparent;'>AI Portfolio Copilot</h1>",
        unsafe_allow_html=True,
    )

    st.caption("Ask about risk, exit strategy, predictions, or portfolio optimization.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container()

    for message in st.session_state.messages:
        with chat_container:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your portfolio…"):

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = generated_response(prompt)
            st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


main()
