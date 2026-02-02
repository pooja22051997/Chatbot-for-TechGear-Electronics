# TechGear Chatbot - Output Execution Steps

This document outlines the step-by-step execution flow of the chatbot application, from data ingestion to query processing.

## 1. Data Ingestion Flow

**Script:** `ingest.py`
**Input:** `data/product_info.txt`
**Output:** ChromaDB Vector Store (`chroma_db/`)

**Steps:**
1.  **Loading**: The text loader reads the raw product data file.
2.  **Splitting**: `RecursiveCharacterTextSplitter` breaks the text into 1000-character chunks with 200-character overlap.
3.  **Embedding**: `GoogleGenerativeAIEmbeddings` (`models/embedding-001`) converts chunks into vector representations.
4.  **Storage**: Vectors are stored in the local ChromaDB `chroma_db` directory.

**Execution Log:**
```text
Loading documents from data/product_info.txt...
Loaded 1 document(s).
Splitting documents (chunk_size=1000, overlap=200)...
Created 41 chunks.
Initializing Google Generative AI Embeddings...
Storing 41 chunks in ChromaDB...
Ingestion complete!
```

---

## 2. Server Startup Flow

**Command:** `uvicorn main:app --reload`
**Output:** FastAPI Server running at `http://0.0.0.0:8000`

**Steps:**
1.  **Environment Check**: Validates `GOOGLE_API_KEY`.
2.  **Database Init**: `auth.py` initializes the SQLite `users.db`.
3.  **Workflow Compilation**: `rag_agent.py` compiles the LangGraph state graph.
4.  **Server Start**: Uvicorn binds to the port and starts listening.

**Execution Log:**
```text
🚀 Starting TechGear Electronics Chatbot API...
✅ User database initialized.
✅ Google API key found.
✅ LangGraph workflow compiled successfully.
✅ API is ready to accept requests.
📚 Swagger docs available at: http://localhost:8000/docs
🌐 Frontend available at: http://localhost:8000
```

---

## 3. Query Processing Flow (LangGraph)

**Endpoint:** `POST /chat`
**Input:** User Query string

The system processes queries through a **LangGraph Workflow**:

### Node 1: Classifier
-   **Input**: User query
-   **Model**: Gemini 2.0 Flash
-   **Action**: Categorizes query into `PRODUCTS`, `RETURNS`, `GENERAL`, or `ESCALATE`.

### Node 2: Routing (Conditional Edge)
-   **If** `ESCALATE` -> Route to **Escalation Node**.
-   **Else** (`PRODUCTS` / `RETURNS` / `GENERAL`) -> Route to **RAG Responder Node**.

### Node 3: Processing
-   **RAG Responder Node**:
    -   Retrieves top k-similar documents from ChromaDB.
    -   Generates answer using Gemini 2.0 Flash with retrieved context.
-   **Escalation Node**:
    -   Returns a pre-defined empathetic message guiding the user to human support.

### Node 4: Response
-   Returns JSON structure: `{ response: "...", category: "...", needs_escalation: bool }`

---

## 4. Example Execution Scenarios

### Scenario A: Product Query
**Input**: "What is the price of SmartWatch Pro X?"
**Flow**: `Classifier (PRODUCTS)` -> `RAG Responder` -> `Response`
**Output**:
```json
{
  "response": "The SmartWatch Pro X is priced at ₹15,999.",
  "category": "PRODUCTS",
  "needs_escalation": false
}
```

### Scenario B: Escalation Query
**Input**: "I want to speak to a manager immediately!"
**Flow**: `Classifier (ESCALATE)` -> `Escalation Node` -> `Response`
**Output**:
```json
{
  "response": "I understand your concern... connecting you with a support specialist.",
  "category": "ESCALATE",
  "needs_escalation": true
}
```
