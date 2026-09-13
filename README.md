# 🚀 AI Chat Assistant with RAG Pipeline

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-4285F4?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203-F55036?style=for-the-badge)

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
## 📂 Project Structure

```text
adrite-agency-platform/
│
├── backend/
│   ├── app/
│   │   ├── ai/
│   │   ├── services/
│   │   ├── utils/
│   │   └── main.py
│   │
│   └── ...
│
├── frontend/
│   └── ...
│
├── docs/
│   └── ...
│
├── .gitignore
├── LICENSE
└── README.md

## ⭐ Project Highlights

| Capability | Technology |
|---|---|
| 🤖 Large Language Model | Groq LLaMA 3 |
| 🔎 Retrieval-Augmented Generation | FAISS |
| 🔤 Text Embeddings | HuggingFace |
| 🧠 AI Orchestration | LangChain |
| ⚡ Backend API | FastAPI |
| 💬 Conversational Context | Last 10 messages |
| 📊 Sentiment Analysis | AI-based analysis |
| 🔮 Intent Prediction | AI-based classification |

## 🧠 Technical Concepts Demonstrated

- **Retrieval-Augmented Generation (RAG)** for knowledge-grounded responses
- **Vector similarity search** using FAISS
- **Text embeddings** using HuggingFace models
- **LLM integration** with Groq LLaMA 3
- **Prompt/context construction** using retrieved knowledge and chat history
- **Conversational memory** using recent messages
- **REST API development** with FastAPI
- **Sentiment analysis** for user messages
- **Intent classification** for query understanding
- **Modular AI application architecture**

## 🔐 Environment Variables

The application requires API credentials for external AI services.

Create a `.env` file in the appropriate project directory and configure the required environment variables.

Example:

```env
GROQ_API_KEY=your_groq_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key

## 📖 API Documentation

The project uses **FastAPI's interactive Swagger UI** for API testing and documentation.

After running the application locally, open:

**Swagger UI:** `http://127.0.0.1:8000/docs`

**ReDoc:** `http://127.0.0.1:8000/redoc`

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

---

## 📄 License

This project is licensed under the **MIT License**.
