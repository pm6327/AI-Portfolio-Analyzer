import streamlit as st 
from utils.chatbot import stream_chat_response
import os 
import json

risk_metrics_path = os.path.join('utils', 'risk_metrics.json')
portfolio_metrics_path = os.path.join('utils', 'portfolio_metrics.json')

# Initialize the context list
def get_user_context():
    context = []

# Load risk metrics
    with open(risk_metrics_path, 'r') as file:
        risk_metrics = json.load(file)
        context.append({"risk_metrics": risk_metrics})

# Load portfolio metrics
    with open(portfolio_metrics_path, 'r') as file:
        portfolio_metrics = json.load(file)
        context.append({"portfolio_metrics": portfolio_metrics})
        
    return context
    
    
def generated_response(prompt):
    context = get_user_context()
    response = stream_chat_response(user_query=prompt,context=context)
    return response



def main():
    st.title(":rainbow[Deep-dive] in your _Portfolio_ here!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container()

    # Render existing chat messages without moving the heading
    for message in st.session_state.messages:
        with chat_container:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        response = ""
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = generated_response(prompt = prompt)
            st.write(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    

main()
    