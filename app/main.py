from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from app.models import AskRequest, AskResponse
from app.rag import ask_question, build_context
from app.embeddings import get_embedding
from app.vector_store import search
from app.reranker import rerank
from app.llm import stream_answer

from app.models import (
    AskRequest,
    AskResponse
)

from app.rag import ask_question


app = FastAPI(
    title="Mini RAG PDF Question Answering",
    description=(
        "PDF Question Answering using "
        "PyMuPDF, Nugen, Qdrant and Reranking"
    ),
    version="1.0.0"
)


@app.get("/")

def root():

    return {
        "message": "Mini RAG API is running"
    }


@app.get("/health")

def health():

    return {
        "status": "ok"
    }


@app.post(
    "/ask",
    response_model=AskResponse
)

def ask(request: AskRequest):

    try:

        result = ask_question(
            request.question
        )

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

@app.post("/ask/stream")
def ask_stream(request: AskRequest):

    try:

        # 1. Question embedding
        query_vector = get_embedding(
            request.question
        )

        # 2. Retrieve Top 10
        retrieved = search(
            query_vector,
            limit=10
        )

        # 3. Rerank Top 10 -> Top 3
        reranked = rerank(
            request.question,
            retrieved
        )

        # 4. Build context
        context = build_context(
            reranked
        )

        # 5. Stream LLM answer
        return StreamingResponse(
            stream_answer(
                request.question,
                context
            ),
            media_type="text/plain"
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )