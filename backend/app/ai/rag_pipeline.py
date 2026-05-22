import os
from typing import List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# LOAD ENV
# =========================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================
# LLM (Groq) ✅ FIXED
# =========================
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",  # ✅ correct model
    temperature=0.3
)

# =========================
# EMBEDDINGS
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# KNOWLEDGE BASE
# =========================
RAW_KNOWLEDGE = [
    "Adrite Agency is a digital solutions company.",
    "Adrite specializes in AI automation, SaaS, and business growth.",
    "We help businesses scale using AI-driven solutions.",
    "Our services include web development, AI chatbots, and automation systems."
]

# =========================
# BUILD VECTOR STORE
# =========================
def build_vector_store():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    docs = []
    for text in RAW_KNOWLEDGE:
        chunks = splitter.split_text(text)
        for chunk in chunks:
            docs.append(Document(page_content=chunk))

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


# ✅ Initialize once
vector_store = build_vector_store()

# =========================
# RETRIEVE CONTEXT
# =========================
def retrieve_context(query: str, k: int = 3) -> List[str]:
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]

# =========================
# GENERATE RESPONSE
# =========================
def generate_response(query: str, context_chunks: List[str]) -> str:
    context = "\n".join(context_chunks)

    prompt = f"""
You are an AI assistant for Adrite Agency.

Use ONLY the context below to answer.

Context:
{context}

User Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content

# =========================
# MAIN RAG PIPELINE
# =========================
def run_rag_pipeline(query: str) -> dict:
    try:
        context_chunks = retrieve_context(query)
        answer = generate_response(query, context_chunks)

        return {
            "success": True,
            "intent": "rag",
            "query": query,
            "context": context_chunks,
            "response": answer
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# =========================
# ENTRY FUNCTION
# =========================
def get_ai_response(message: str, client_id: int = None, chat_history: list = None):
    try:
        # =========================
        # FORMAT CHAT HISTORY
        # =========================
        history_text = ""

        if chat_history:
            for chat in chat_history:

                # ✅ Format 1: {message, response}
                if "message" in chat and "response" in chat:
                    user_msg = chat.get("message", "")
                    ai_msg = chat.get("response", "")

                    history_text += f"User: {user_msg}\n"
                    history_text += f"AI: {ai_msg}\n"

                # ✅ Format 2: {role, content}
                elif "role" in chat and "content" in chat:
                    if chat["role"] == "user":
                        history_text += f"User: {chat['content']}\n"
                    elif chat["role"] == "assistant":
                        history_text += f"AI: {chat['content']}\n"

        # =========================
        # GET RAG CONTEXT
        # =========================
        context_chunks = retrieve_context(message)
        context = "\n".join(context_chunks)

        # =========================
        # FINAL PROMPT
        # =========================
        prompt = f"""
You are an AI assistant for Adrite Agency.

Use the context and chat history to answer.

Chat History:
{history_text}

Context:
{context}

User Question:
{message}

Answer:
"""

        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        print("RAG ERROR:", str(e))
        return "Sorry, something went wrong."