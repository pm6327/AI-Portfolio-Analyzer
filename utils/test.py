from crew import CrewaiConversationalChatbotCrew
from langchain.memory import ConversationBufferMemory
from mongo_manager import MongoManager
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")
LLM =ChatGroq(api_key=groq_api, model="mixtral-8x7b-32768")

LLM = CrewaiConversationalChatbotCrew(llm={'provider': 'Groq', 'model': 'mixtral-8x7b-32768'})
# Initialize ConversationBufferMemory
memory = ConversationBufferMemory()

#  Connect to MongoDB and retrieve the starting context
# def get_initial_context_from_mongo():
#     mongo_db_1 =MongoManager(collection_name="portfolio_metrics",db_name="portfolio_data")
#     mongo_db_2 =MongoManager(collection_name="risk_metrics",db_name="portfolio_data")
    
#     # Fetch the context document, e.g., the latest one or a specific one
#     document_1 = mongo_db_1.find_documents({"_id": "context_document_id"})  # Update with appropriate query
#     document_2 =mongo_db_2.find_documents({})
#     initial_context = document.get("initial_context", "") if document else ""
#     mongo_db_2.close_connection
#     mongo_db_1.close_connection
#     return initial_context

def run():
    # Load initial context from MongoDB
    # initial_context = get_initial_context_from_mongo()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye! It was nice talking to you.")
            break

        # Retrieve conversation history from LangChain memory
        conversation_history = memory.load_memory_variables({}).get("history", "")
        
        # Combine MongoDB initial context with conversation history
        context = f"{conversation_history}".strip()

        # Prepare inputs for the chatbot with user input and combined context
        inputs = {
            "user_message": user_input,
            "context": context
        }

        # Get chatbot response
        response = CrewaiConversationalChatbotCrew(llm={'provider': 'Groq', 'model': 'mixtral-8x7b-32768'}).crew().kickoff(inputs=inputs)


        # Save both the user input and chatbot response to memory
        memory.save_context({"input": user_input}, {"output": response})
        print(f"Assistant: {response}")

# Run the chat function
run()
