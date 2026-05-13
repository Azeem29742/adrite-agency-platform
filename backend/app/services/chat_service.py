import os
import requests
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from app.services.rag_service import get_rag_response

load_dotenv()

# Initialize LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)


# ✅ Intent Detection
def detect_intent(query: str):
    query = query.lower()

    if "ticket" in query or "sentiment" in query:
        return "sentiment"

    elif "predict" in query or "forecast" in query:
        return "prediction"

    else:
        return "rag"


# ✅ External API Calls

def call_sentiment_api():
    try:
        response = requests.get("http://localhost:8001/api/sentiment")
        return response.json()
    except:
        return {"error": "Sentiment service not available"}


def call_prediction_api():
    try:
        response = requests.get("http://localhost:8002/api/predict")
        return response.json()
    except:
        return {"error": "Prediction service not available"}


# ✅ Main Chat Function
def get_ai_response(messages: list):
    try:
        # 🔹 Extract latest user query
        user_query = messages[-1]["content"]

        # 🔹 Detect intent
        intent = detect_intent(user_query)

        # ===============================
        # 🔥 ROUTING LOGIC (CORE FEATURE)
        # ===============================

        if intent == "sentiment":
            data = call_sentiment_api()

            if "error" in data:
                return "Sentiment service is currently unavailable."

            return f"📊 Sentiment Analysis Result:\n{data}"

        elif intent == "prediction":
            data = call_prediction_api()

            if "error" in data:
                return "Prediction service is currently unavailable."

            return f"📈 Prediction Result:\n{data}"

        else:
            # 🔹 RAG + LLM (default flow)
            context = get_rag_response(user_query)

            system_prompt = f"""
You are an AI assistant for Adrite Agency.

Use the following context to answer the user:

{context}

If the answer is not in the context, say politely you don't know.
"""

            formatted_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]

            response = llm.invoke(formatted_messages)

            return response.content

    except Exception as e:
        return f"Error: {str(e)}"