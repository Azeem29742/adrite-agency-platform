# 🚀 AI Chat Assistant with RAG Pipeline

An AI-powered backend system built using FastAPI, integrating Retrieval-Augmented Generation (RAG) with chat history for contextual and intelligent responses.

---

## 🔥 Key Features

- 💬 Chat API with context-aware responses
- 🧠 RAG pipeline using FAISS + HuggingFace embeddings
- 📚 Knowledge-based responses (Adrite Agency)
- 🔁 Chat history integration (last 10 messages)
- 📊 Sentiment Analysis API
- 🔮 Intent Prediction API
- ⚡ FastAPI backend with structured responses
- 🛠 Clean architecture (services, utils, ai modules)

---

## 🏗 Architecture



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