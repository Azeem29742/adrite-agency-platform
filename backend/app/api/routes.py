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

        # ✅ Extract latest user query
        query = messages[-1]["content"]

        # ✅ Detect intent (Step 2)
        intent = detect_intent(query)

        # ✅ Generate AI response
        ai_response = get_ai_response(messages)

        return {
            "success": True,
            "message": "Response generated successfully",
            "data": {
                "intent": intent,  # (optional but good for debugging/demo)
                "conversation": messages,
                "ai_response": ai_response
            }
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e),
            "data": None
        }
    
# -----------------------------
# New Request Schema for AI APIs
# -----------------------------
class TextRequest(BaseModel):
    text: str


# -----------------------------
# Sentiment API (NEW)
# -----------------------------
@router.post("/sentiment")
def sentiment_analysis(request: TextRequest):
    try:
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
        return {
            "success": False,
            "message": str(e),
            "data": None
        }


# -----------------------------
# Prediction API (NEW)
# -----------------------------
@router.post("/predict")
def prediction(request: TextRequest):
    try:
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
        return {
            "success": False,
            "message": str(e),
            "data": None
        }
    
@router.get("/check")
def check():
    return {"status": "ok"}