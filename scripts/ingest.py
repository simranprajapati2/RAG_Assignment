import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from app.pdf_processor import (
    extract_text_from_pdf,
    create_chunks_from_pages
)

from app.embeddings import get_embedding
from app.vector_store import clear_collection, store_chunks


PDF_PATH = "data/sample.pdf"


def main():

    print("\n")
    print("=" * 70)
    print("MINI RAG DOCUMENT INGESTION")
    print("=" * 70)

    # --------------------------------------------------
    # STEP 1: PDF PROCESSING
    # --------------------------------------------------

    print("\nSTEP 1: PDF PROCESSING")

    if not os.path.exists(PDF_PATH):
        print("ERROR: PDF not found:", PDF_PATH)
        return

    print("Reading:", PDF_PATH)

    pages = extract_text_from_pdf(PDF_PATH)

    print("Pages extracted:", len(pages))

    for page in pages:
        print(
            f"Page {page['page']} -> "
            f"{len(page['text'])} characters"
        )

    # --------------------------------------------------
    # SHOW ACTUAL PDF CONTENT
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CHECKING PDF CONTENT")
    print("=" * 70)

    for page in pages[:3]:

        print(f"\n--- PAGE {page['page']} ---")

        print(page["text"][:1000])

    # --------------------------------------------------
    # STEP 2: CHUNKING
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 2: CHUNKING")
    print("=" * 70)

    chunks = create_chunks_from_pages(
        pages,
        chunk_size=800,
        chunk_overlap=150
    )

    print("Total chunks:", len(chunks))

    for index, chunk in enumerate(chunks[:10], start=1):

        print(f"\nChunk {index}")
        print("Page:", chunk["page"])
        print("Text:", chunk["text"][:300])

    # --------------------------------------------------
    # STEP 3: EMBEDDINGS
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 3: GENERATING EMBEDDINGS")
    print("=" * 70)

    embeddings = []

    for index, chunk in enumerate(chunks, start=1):

        embedding = get_embedding(chunk["text"])

        embeddings.append(embedding)

        print(
            f"Embedded {index}/{len(chunks)} "
            f"| dimension={len(embedding)}"
        )

    # --------------------------------------------------
    # STEP 4: QDRANT
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 4: STORING IN QDRANT")
    print("=" * 70)

    clear_collection()

    stored = store_chunks(
        embeddings,
        chunks
    )

    print("\nStored chunks:", stored)

    # --------------------------------------------------
    # COMPLETE
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()