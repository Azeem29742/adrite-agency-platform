import os
import requests
from dotenv import load_dotenv

from langchain_groq import ChatGroq

# 👉 Import YOUR pipeline (IMPORTANT)
from app.ai.rag_pipeline import get_ai_response as rag_pipeline_response

load_dotenv()

# Initialize LLM (still used for other flows if needed)
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-70b-8192"
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
        return {"error": "Prediction service not available. Please try again later."}


# ✅ Main Chat Function
def get_ai_response(messages: list):
    try:
        # 🔹 Extract latest user query
        user_query = messages[-1]["content"]

        # 🔹 Detect intent
        intent = detect_intent(user_query)

        # ===============================
        # 🔥 ROUTING LOGIC
        # ===============================

        if intent == "sentiment":
            data = call_sentiment_api()

            if "error" in data:
                return "Sentiment service is currently unavailable. Please try again later."

            return f"📊 Sentiment Analysis Result:\n{data}"

        elif intent == "prediction":
            data = call_prediction_api()

            if "error" in data:
                return "Prediction service is currently unavailable."

            return f"📈 Prediction Result:\n{data}"

        else:
           return rag_pipeline_response(user_query)

    except Exception as e:
        return f"Error: {str(e)}"