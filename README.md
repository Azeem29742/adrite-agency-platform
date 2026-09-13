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

## 🛠 Tech Stack

### Backend
- 🐍 **Python**
- ⚡ **FastAPI**
- 🔗 **REST APIs**

### AI & LLM
- 🧠 **LangChain**
- 🤖 **Groq LLM (LLaMA 3)**
- 🔤 **HuggingFace Embeddings**

### RAG & Vector Search
- 🔎 **Retrieval-Augmented Generation (RAG)**
- 📚 **FAISS Vector Search**

---

## 📡 API Endpoints

| Feature | Method | Endpoint |
|---|---|---|
| 💬 Chat | `POST` | `/api/v1/chat/` |
| 📊 Sentiment Analysis | `POST` | `/sentiment` |
| 🔮 Intent Prediction | `POST` | `/predict` |

### 💬 Chat API

Handles context-aware conversations using:

- User message
- Previous chat history
- Retrieved RAG context
- LLM-generated response

### 📊 Sentiment Analysis API

Analyzes the sentiment of the user's message.

### 🔮 Intent Prediction API

Predicts the intent behind the user's query.
---

## 🧠 How It Works

The system follows a RAG-based workflow to generate context-aware responses:

1. **User sends a query** through the Chat API.
2. **Chat history is retrieved** to maintain conversational context.
3. **The query is processed** and used to search the knowledge base.
4. **FAISS retrieves relevant information** using HuggingFace embeddings.
5. **Retrieved context and recent conversation history** are combined with the user's query.
6. **Groq LLaMA 3** processes the combined context.
7. **The generated response** is returned through the FastAPI endpoint.

### 🔄 RAG Flow

```text
User Query
    ↓
Query Processing
    ↓
Embedding Generation
    ↓
FAISS Similarity Search
    ↓
Relevant Context
    +
Chat History
    ↓
Groq LLaMA 3
    ↓
Context-Aware Response

---

## ▶️ Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Azeem29742/adrite-agency-platform.git
cd adrite-agency-platform
2. Create a Virtual Environment
python -m venv venv

Windows:

venv\Scripts\activate

macOS/Linux:

source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Configure Environment Variables

Create a .env file in the project root and add the required API credentials.

Never commit API keys or other secrets to GitHub.

5. Start the FastAPI Server
uvicorn app.main:app --reload

The API will be available at:

http://127.0.0.1:8000
📚 Interactive API Documentation

Once the server is running, open:

http://127.0.0.1:8000/docs

## 📌 Project Status

🟢 **Active Development**

The core AI backend, RAG pipeline, conversational context, sentiment analysis, and intent prediction features are implemented.

### 🔮 Future Improvements

- 🔐 Add authentication and authorization
- 🗃️ Expand and improve the knowledge base
- 🧪 Add automated tests
- 📈 Add monitoring and logging
- 🚀 Prepare the application for production deployment
- 💬 Improve conversational memory and response quality
