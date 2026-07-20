# 🤖 Local Production-Ready LangGraph RAG Assistant

A fully containerized, local Retrieval-Augmented Generation (RAG) and Agentic workflow pipeline built with **LangGraph**, **FastAPI**, **ChromaDB**, **Ollama (Llama 3.1)**, and **Streamlit**. 

This system features stateful agentic routing, hybrid retrieval with cross-encoder re-ranking, real-time token-by-token streaming over Server-Sent Events (SSE), and a built-in benchmark evaluation suite.

---

## 🌟 Key Features

* **100% Local Inference**: Runs `Llama 3.1` locally via Ollama with zero external LLM API costs or privacy concerns.
* **Stateful Workflow Engine**: Utilizes **LangGraph** (`StateGraph`) to dynamically route queries between Local Document RAG, Direct Answers, and External Web Search.
* **Hybrid Retrieval & Re-ranking**: Combines dense vector search with sparse keyword search (BM25) and Cross-Encoder re-ranking (`sentence-transformers`) for high-precision context extraction.
* **Real-time SSE Streaming**: Endpoint `/chat` streams response tokens via Server-Sent Events with custom anti-buffering headers.
* **Interactive UI**: Feature-rich Streamlit frontend with interactive document upload, sidebar controls, and live streaming conversation history.
* **Automated Benchmark Evaluation**: Built-in evaluation pipeline (`eval/evaluate.py`) measuring route accuracy, answer relevance, and context precision against benchmark datasets.

---

## 🏗️ Architecture & Technology Stack

```text
                     +---------------------------------------+
                     |         Streamlit Frontend            |
                     |         (Port 8501)                   |
                     +-------------------+-------------------+
                                         |
                                         v
                     +-------------------+-------------------+
                     |          FastAPI Backend              |
                     |          (Port 8000)                  |
                     +---------+-----------------+-----------+
                               |                 |
            +------------------+                 +------------------+
            |                                                       |
            v                                                       v
+-----------+-----------+                               +-----------+-----------+
|    ChromaDB Vector DB |                               |  Ollama (Llama 3.1)   |
|    (Port 8001)        |                               |  (Port 11434)         |
+-----------------------+                               +-----------------------+
```

* **Backend**: FastAPI, Uvicorn, LangChain, LangGraph
* **Frontend**: Streamlit
* **Vector Store**: ChromaDB
* **LLM Engine**: Ollama (`llama3.1`)
* **Re-ranker / Embeddings**: HuggingFace Cross-Encoders, PyTorch, Sentence-Transformers
* **Web Search Tool**: Tavily API

---

## 📂 Project Structure

```text
Ai-assistant/
├── app/
│   ├── api/                # FastAPI routes (/chat, /upload)
│   ├── main.py             # App entry point
│   └── rag/
│       ├── agent.py        # LangGraph workflow definition & routing logic
│       └── vectorstore.py  # ChromaDB & Hybrid Search initialization
├── eval/
│   ├── test_dataset.json   # Golden evaluation benchmark dataset
│   └── evaluate.py         # Evaluation run script
├── frontend/
│   ├── app.py              # Streamlit UI
│   ├── Dockerfile          # Frontend container definition
│   └── requirements.txt    # Frontend dependencies
├── Dockerfile              # Backend multi-stage build container definition
├── docker-compose.yaml     # Orchestration file for full local stack
├── test_agent.py           # CLI script for testing agent routing
├── requirements.txt        # Backend dependencies
└── README.md
```

---

## ⚙️ Prerequisites

Before running the project locally, ensure you have installed:
* **Docker** (v20.10 or higher)
* **Docker Compose** (v2.0 or higher)
* **Git**
* Minimum Hardware: **16 GB RAM** recommended to run Llama 3.1 smoothly alongside ChromaDB.

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone [https://github.com/gowthamgoshike/Ai-assistant.git](https://github.com/gowthamgoshike/Ai-assistant.git)
cd Ai-assistant
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:

```env
# Tavily Web Search API (Optional: required only for web routing)
TAVILY_API_KEY=your_tavily_api_key_here

# HuggingFace Token (Optional: prevents rate-limit warnings when downloading models)
HF_TOKEN=your_huggingface_token_here
```

### 3. Spin Up the Docker Stack
Run the following command from the project root:

```bash
docker-compose up -d --build
```

Docker Compose will build and spin up 5 services:
1. `rag_backend`: FastAPI server running at `http://localhost:8000`
2. `rag_frontend`: Streamlit app running at `http://localhost:8501`
3. `rag_vector_db`: ChromaDB instance running at `http://localhost:8001`
4. `ollama`: Ollama server running at `http://localhost:11434`
5. `ollama_init`: Transient initialization service that automatically pulls `llama3.1` into the Ollama container volume on first run.

---

## 💻 Accessing the Application

* **Streamlit Web UI**: Navigate to `http://localhost:8501` in your browser to interact with the assistant and ingest PDF documents.
* **Interactive API Documentation (Swagger)**: Visit `http://localhost:8000/docs` to view and test backend REST endpoints.
* **ChromaDB Heartbeat**: Access `http://localhost:8001/api/v1/heartbeat` to verify vector database health.

---

## 🧪 Testing & Evaluation Suite

### Direct Terminal Agent Testing
To quickly test route execution (`rag` vs `web_node`) directly via terminal:

```bash
python test_agent.py
```

### Running Benchmark Evaluation
To execute performance benchmarking and routing evaluation inside the containerized environment:

```bash
docker exec -it rag_backend python -m eval.evaluate
```

This will run the agent against `eval/test_dataset.json` and generate an output CSV file (`eval/eval_results.csv`) detailing route accuracy, generated answers, and retrieved contexts.

---

## 🔧 Useful Docker Commands

* **View Ollama Model Download Progress**:
  ```bash
  docker logs -f ollama
  ```
* **View Backend Logs**:
  ```bash
  docker logs -f rag_backend
  ```
* **Stop the Stack**:
  ```bash
  docker-compose down
  ```
* **Stop and Purge Persistent Volumes** (Wipes ChromaDB & Ollama caches):
  ```bash
  docker-compose down -v
  ```
