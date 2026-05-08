import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from app.services.rag_service import get_rag_response

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

def get_ai_response(messages: list):
    try:
        formatted_messages = []

        # Get latest user message
        user_query = messages[-1]["content"]

        # 🔥 Get RAG context
        context = get_rag_response(user_query)

        # System prompt with context
        formatted_messages.append(
            SystemMessage(
                content=f"""
You are an AI assistant for Adrite Agency.

Use the following context to answer the user:

{context}

If the answer is not in the context, say politely you don't know.
"""
            )
        )

        # Add user message
        formatted_messages.append(
            HumanMessage(content=user_query)
        )

        response = llm.invoke(formatted_messages)

        return response.content

    except Exception as e:
        return f"Error: {str(e)}"