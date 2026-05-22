from app.utils.response import success_response, error_response
from app.services.ai_service import analyze_sentiment, predict_intent
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

from app.services.chat_service import get_ai_response, detect_intent
import app.services.ai_service as ai_service

router = APIRouter()


# Message structure (for conversation)
class Message(BaseModel):
    role: str
    content: str


# Request model (supports full conversation)
class ChatRequest(BaseModel):
    messages: List[Message]


@router.post("/chat/send")
def chat(request: ChatRequest):
    try:
        # Convert Pydantic objects to dict
        messages = [msg.dict() for msg in request.messages]

        # ✅ Input validation
        if not messages:
            raise ValueError("Messages list is empty")

        if "content" not in messages[-1] or not messages[-1]["content"].strip():
            raise ValueError("User message is empty")

        # ✅ Extract latest user query
        query = messages[-1]["content"]

        # ✅ Detect intent
        intent = detect_intent(query)

        # ✅ Generate AI response
        ai_response = get_ai_response(messages)

        return success_response(
        "Response generated successfully",
        {"answer": ai_response}
)

    except ValueError as ve:
        return error_response(str(ve))

    except Exception as e:
        print("ERROR:", str(e))  # ✅ internal log

    return {
        "success": False,
        "message": "Something went wrong. Please try again.",
        "data": None
    }


# -----------------------------
# New Request Schema for AI APIs
# -----------------------------
class TextRequest(BaseModel):
    text: str


# -----------------------------
# Sentiment API
# -----------------------------
@router.post("/sentiment")
def sentiment_analysis(request: TextRequest):
    try:
        if not request.text.strip():
            raise ValueError("Text cannot be empty")

        sentiment = ai_service.analyze_sentiment(request.text)

        return {
            "success": True,
            "message": "Sentiment analyzed successfully",
            "data": {
                "text": request.text,
                "sentiment": sentiment
            }
        }

    except Exception as e:
        print("ERROR:", str(e))

        return {
            "success": False,
            "message": "Something went wrong. Please try again.",
            "data": None
        }


# -----------------------------
# Prediction API
# -----------------------------
@router.post("/predict")
def prediction(request: TextRequest):
    try:
        if not request.text.strip():
            raise ValueError("Text cannot be empty")

        prediction = ai_service.predict_intent(request.text)

        return {
            "success": True,
            "message": "Prediction generated successfully",
            "data": {
                "text": request.text,
                "prediction": prediction
            }
        }

    except Exception as e:
        print("ERROR:", str(e))

        return {
            "success": False,
            "message": "Something went wrong. Please try again.",
            "data": None
        }


@router.get("/check")
def check():
    return {"status": "ok"}