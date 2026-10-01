from sentence_transformers import CrossEncoder
from app.config import settings
import re


print("\nLoading reranker model...")
reranker = CrossEncoder(settings.RERANKER_MODEL)
print("Reranker loaded.")


def normalize(text: str):
    text = text.lower()
    text = text.replace("syntaxerror", "syntax error")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def keyword_score(question: str, document: str):
    question_words = set(normalize(question).split())
    document_words = set(normalize(document).split())

    if not question_words:
        return 0.0

    common_words = question_words.intersection(document_words)

    return len(common_words) / len(question_words)


def rerank(question: str, documents: list[dict]):

    if not documents:
        return []

    # CrossEncoder score
    pairs = [
        (question, document["text"])
        for document in documents
    ]

    cross_scores = reranker.predict(pairs)

    reranked = []

    for document, cross_score in zip(
        documents,
        cross_scores
    ):

        item = document.copy()

        # Original semantic score
        semantic_score = float(
            document.get("score", 0.0)
        )

        # Cross encoder score
        cross_score = float(cross_score)

        # Keyword score
        keyword = keyword_score(
            question,
            document["text"]
        )

        # Combined score
        final_score = (
            (0.45 * cross_score)
            + (0.35 * semantic_score)
            + (0.20 * keyword)
        )

        item["cross_score"] = cross_score
        item["keyword_score"] = keyword
        item["rerank_score"] = final_score

        reranked.append(item)

    reranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked[:settings.RERANK_TOP_K]