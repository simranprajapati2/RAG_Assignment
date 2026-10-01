# Mini RAG PDF Question Answering

A minimal Retrieval-Augmented Generation (RAG) API built with FastAPI, Qdrant, PyMuPDF, Sentence Transformers, and Nugen as the inference and embedding provider.

The application processes PDF documents, creates text chunks, generates embeddings, stores them in Qdrant, retrieves relevant chunks, reranks them, and generates answers using the Nugen LLM.

## Stack

| **Layer** | **Technology** |
| ---------- | --------------- |
| API | FastAPI |
| PDF Processing | PyMuPDF |
| Vector DB | Qdrant |
| Similarity | Cosine |
| Embeddings | Nugen API |
| Reranking | Sentence Transformers Cross-Encoder |
| LLM | Nugen API |
| HTTP Client | HTTPX |
| Validation | Pydantic |

## Project Structure

```text
mini-rag/
├── app/
│   ├── __init__.py
│   ├── config.py                  # Application configuration
│   ├── models.py                  # Pydantic request/response models
│   ├── pdf_processor.py           # PDF extraction and chunking
│   ├── embeddings.py              # Nugen embedding client
│   ├── vector_store.py            # Qdrant vector store
│   ├── reranker.py                # Cross-Encoder + keyword reranking
│   ├── llm.py                     # Nugen LLM client + streaming
│   ├── rag.py                     # RAG pipeline orchestration
│   └── main.py                    # FastAPI application
│
├── scripts/
│   └── ingest.py                  # PDF ingestion script
│
├── data/
│   ├── sample.pdf                 # Sample PDF document
│   └── qdrant/                    # Local Qdrant storage
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md


## Setup
# 1. Clone and enter the project

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment

# Windows
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure environment
# Create a .env file and add your Nugen API key

## Then run ingestion:

python scripts/ingest.py

After ingestion is complete, restart the API:

uvicorn app.main:app --reload

Open Swagger

http://127.0.0.1:8000/docs

Test /ask

Test /ask/stream

