import streamlit as st
from utils.chatbot import stream_chat_response
import os
import json


# ---------- PAGE CONFIG ----------
st.set_page_config(layout="wide")

st.markdown(
    """
<style>
.header-card {
    padding: 18px;
    border-radius: 14px;
    background: linear-gradient(135deg,#020617,#020617);
    border: 1px solid #1e293b;
}
.chat-hint {
    padding: 12px;
    border-radius: 10px;
    background-color: #020617;
    border: 1px solid #1e293b;
    margin-bottom: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================
# Paths
# =============================
risk_metrics_path = os.path.join("utils", "risk_metrics.json")
portfolio_metrics_path = os.path.join("utils", "portfolio_metrics.json")
prediction_path = os.path.join("utils", "model_predictions.json")


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
# Intent Detection
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
# UI LAYOUT
# =============================

# ----- HEADER
st.markdown(
    """
<div class="header-card">
<h2>🤖 AI Portfolio Copilot</h2>
<p>Your intelligent assistant for risk insights, exits, rebalancing and future forecasts.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ----- SUGGESTED PROMPTS
st.markdown("### 💬 Ask things like")

hint1, hint2, hint3, hint4 = st.columns(4)

hint1.markdown(
    '<div class="chat-hint">Should I rebalance my portfolio?</div>',
    unsafe_allow_html=True,
)
hint2.markdown(
    '<div class="chat-hint">Which stock is riskiest right now?</div>',
    unsafe_allow_html=True,
)
hint3.markdown(
    '<div class="chat-hint">Any stock I should sell?</div>', unsafe_allow_html=True
)
hint4.markdown(
    '<div class="chat-hint">Future outlook for my portfolio</div>',
    unsafe_allow_html=True,
)


# =============================
# CHAT SYSTEM
# =============================

if "messages" not in st.session_state:
    st.session_state.messages = []

chat_container = st.container()

for message in st.session_state.messages:
    with chat_container:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# ----- USER INPUT
if prompt := st.chat_input("Ask about your portfolio…"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Analyzing portfolio and generating insights..."):
            response = generated_response(prompt)

        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
