from fastapi import APIRouter
from pydantic import BaseModel

from app.services.chat_service import get_ai_response

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat/send")
def chat(request: ChatRequest):

    ai_response = get_ai_response(request.message)

    return {
        "response": ai_response
    }