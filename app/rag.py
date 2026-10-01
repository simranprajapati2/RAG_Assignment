from app.embeddings import get_embedding
from app.vector_store import search
from app.reranker import rerank
from app.llm import generate_answer


def build_context(documents):

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        context_parts.append(
            f"""
Document {index}

Source: {document["source"]}

Page: {document["page"]}

Text:
{document["text"]}
"""
        )

    return "\n---\n".join(context_parts)


def ask_question(question: str):

    print("\n")
    print("=" * 60)
    print("RAG PIPELINE STARTED")
    print("=" * 60)

    # ------------------------------------------------
    # STEP 1
    # ------------------------------------------------

    print("\nSTEP 1: Creating question embedding...")

    query_vector = get_embedding(question)

    # ------------------------------------------------
    # STEP 2
    # ------------------------------------------------

    print("\nSTEP 2: Retrieving Top 10...")

    retrieved = search(
        query_vector,
        limit=10
    )

    print(
        f"\nRetrieved {len(retrieved)} chunks"
    )

    retrieved_response = []

    print("\n--- TOP 10 BEFORE RERANKING ---")

    for index, document in enumerate(
        retrieved,
        start=1
    ):

        print(
            f"{index}. "
            f"Page={document['page']} "
            f"Score={document['score']:.4f}"
        )

        retrieved_response.append({
            "rank": index,
            "page": document["page"],
            "source": document["source"],
            "score": document["score"],
            "text": document["text"]
        })

    # ------------------------------------------------
    # STEP 3
    # ------------------------------------------------

    print("\nSTEP 3: Reranking Top 10...")

    reranked = rerank(
        question,
        retrieved
    )

    reranked_response = []

    print("\n--- TOP 3 AFTER RERANKING ---")

    for index, document in enumerate(
        reranked,
        start=1
    ):

        print(
            f"{index}. "
            f"Page={document['page']} "
            f"Rerank Score="
            f"{document['rerank_score']:.4f}"
        )

        print(
            f"   Keyword Score="
            f"{document['keyword_score']:.4f}"
        )

        reranked_response.append({
            "rank": index,
            "page": document["page"],
            "source": document["source"],
            "score": document["rerank_score"],
            "text": document["text"]
        })

    # ------------------------------------------------
    # STEP 4
    # ------------------------------------------------

    print("\nSTEP 4: Building prompt context...")

    context = build_context(reranked)

    # ------------------------------------------------
    # STEP 5
    # ------------------------------------------------

    print("\nSTEP 5: Calling Nugen LLM...")

    answer = generate_answer(
        question,
        context
    )

    # ------------------------------------------------
    # STEP 6
    # ------------------------------------------------

    print("\nSTEP 6: FINAL ANSWER")

    print(answer)

    print("\n")
    print("=" * 60)
    print("RAG PIPELINE COMPLETED")
    print("=" * 60)

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": retrieved_response,
        "reranked_chunks": reranked_response
    }