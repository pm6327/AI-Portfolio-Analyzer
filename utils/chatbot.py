from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")
if not groq_api:
    raise ValueError("GROQ_API_KEY not found")

chat_model = ChatGroq(
    api_key=groq_api,
    model="llama-3.3-70b-versatile"
)

SYSTEM_PROMPT = """As an experienced AI financial assistant, I bring extensive knowledge in stock market analysis,
risk management, and portfolio strategy.

Context:
{metrics}
"""

def stream_chat_response(context, user_query):
    system_prompt_filled = SYSTEM_PROMPT.format(metrics=context)

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt_filled),
        HumanMessage(content=user_query)
    ])

    messages = chat_prompt.format_messages()
    response = chat_model.invoke(messages)

    return response.content
