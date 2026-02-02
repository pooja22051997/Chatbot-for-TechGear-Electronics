# TechGear Electronics Customer Support Chatbot 🤖

A RAG-based (Retrieval Augmented Generation) customer support chatbot powered by **Google Gemini 2.0 Flash**, **LangChain**, **LangGraph**, and **FastAPI**.

![Chatbot UI](https://via.placeholder.com/800x400?text=TechGear+Chatbot+UI)

## 🌟 Features

-   **Intelligent Query Classification**: Automatically routes queries to Products, Returns, General, or Human Escalation.
-   **RAG Knowledge Base**: Uses ChromaDB to retrieve accurate product info from a local dataset.
-   **Human Escalation**: Detects angry or complex queries and routes them to a human agent flow.
-   **Secure Authentication**: JWT-based Signup/Login system with SQLite database.
-   **Modern UI**: Dark-themed, responsive chat interface with sidebar navigation.
-   **API Documentation**: Interactive Swagger UI for testing endpoints.

## 🛠️ Tech Stack

-   **LLM**: Google Gemini 2.0 Flash
-   **Orchestration**: LangChain & LangGraph
-   **Vector DB**: ChromaDB
-   **Backend**: FastAPI (Python)
-   **Frontend**: HTML/CSS/JS (Vanilla)
-   **Auth**: JWT (Python-Jose) + Bcrypt

## 🚀 Setup & Installation

### Prerequisites
-   Python 3.10+
-   Google Cloud API Key (for Gemini)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/techgear-chatbot.git
cd techgear-chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```bash
cp .env.example .env
```
Edit `.env` and add your Google API key:
```ini
GOOGLE_API_KEY=your_google_api_key_here
SECRET_KEY=your_secret_key_here
```

### 5. Ingest Knowledge Base
Load the product data into ChromaDB:
```bash
python ingest.py
```

## 🏃‍♂️ Running the Application

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

-   **Frontend**: Open [http://localhost:8000](http://localhost:8000)
-   **Swagger Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs)

## 📁 Project Structure

```
.
├── auth.py             # Authentication logic (JWT, SQLite)
├── chroma_db/          # Vector database storage
├── data/
│   └── product_info.txt # Knowledge base source file
├── ingest.py           # Script to ingest data into ChromaDB
├── main.py             # FastAPI entry point & endpoints
├── rag_agent.py        # LangGraph workflow definition
├── rag_chain.py        # RAG implementation with Gemini
├── static/
│   └── index.html      # Frontend Application
├── users.db            # SQLite user database
└── requirements.txt    # Project dependencies
```

## 🧪 Testing

### Sample Queries
-   **Product**: "What is the price of SmartWatch Pro X?"
-   **Policy**: "What is your return policy?"
-   **General**: "Do you have a physical store?"
-   **Escalation**: "I want to speak to a manager, this is unacceptable!"

### API Endpoints
-   `POST /auth/signup`: Register new user
-   `POST /auth/login`: Login and get JWT token
-   `POST /chat`: Protected chat endpoint (requires Bearer token)
-   `GET /health`: System health check

## 📝 License

This project is licensed under the MIT License.
