# 🚀 AI Chat Assistant with RAG Pipeline

An AI-powered backend system built with **FastAPI** that combines **Retrieval-Augmented Generation (RAG)**, conversational chat history, sentiment analysis, and intent prediction to deliver contextual and intelligent responses for Adrite Agency.

---

## 🔥 Key Features

- 💬 **Context-Aware Chat API** — Generates responses using conversation context and retrieved knowledge.
- 🧠 **RAG Pipeline** — Retrieves relevant information from the knowledge base using FAISS and HuggingFace embeddings.
- 📚 **Agency Knowledge Base** — Provides responses grounded in Adrite Agency-specific information.
- 🔁 **Conversation History** — Incorporates the last 10 messages to maintain conversational context.
- 📊 **Sentiment Analysis** — Analyzes the emotional tone of incoming user messages.
- 🔮 **Intent Prediction** — Identifies the likely intent behind user queries.
- ⚡ **FastAPI Backend** — Provides structured, scalable REST API endpoints.
- 🛠️ **Modular Architecture** — Separates AI logic, services, utilities, and API components for maintainability.

---

## 🏗 Architecture

## 🏗 Architecture

```text
                    ┌──────────────────┐
                    │    User Query   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   FastAPI Chat   │
                    │       API        │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
     ┌─────────────────┐           ┌─────────────────┐
     │  Chat History   │           │  RAG Retrieval  │
     │  Last 10 Msgs   │           │     FAISS       │
     └────────┬────────┘           └────────┬────────┘
              │                             │
              │                    ┌────────▼────────┐
              │                    │ HuggingFace     │
              │                    │   Embeddings    │
              │                    └────────┬────────┘
              │                             │
              └──────────────┬──────────────┘
                             ▼
                    ┌──────────────────┐
                    │   Context +      │
                    │  User Question   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Groq LLaMA 3    │
                    │       LLM        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  AI Response     │
                    └──────────────────┘

---

## 🛠 Tech Stack

- Python
- FastAPI
- LangChain
- Groq LLM (LLaMA 3)
- FAISS
- HuggingFace Embeddings

---

## 📡 API Endpoints

### Chat
POST `/api/v1/chat/`

### Sentiment
POST `/sentiment`

### Prediction
POST `/predict`

---

## 🧠 How It Works

1. User sends message
2. System retrieves relevant context (FAISS)
3. Combines:
   - Chat history
   - Retrieved context
4. LLM generates response

---

## ▶️ Run Locally

```bash
uvicorn app.main:app --reload
