from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

from app.services.chat_service import get_ai_response

router = APIRouter()


# Message structure (for conversation)
class Message(BaseModel):
    role: str
    content: str


# Request model (now supports full conversation)
class ChatRequest(BaseModel):
    messages: List[Message]


@router.post("/chat/send")
def chat(request: ChatRequest):
    try:
        # Convert Pydantic objects to dict (IMPORTANT)
        messages = [msg.dict() for msg in request.messages]

        ai_response = get_ai_response(messages)

        return {
            "success": True,
            "message": "Response generated successfully",
            "data": {
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