import pymupdf
import re


def extract_text_from_pdf(pdf_path: str):
    document = pymupdf.open(pdf_path)

    pages = []

    for page_number in range(len(document)):
        page = document[page_number]

        text = page.get_text("text")

        if text and text.strip():
            pages.append({
                "page": page_number + 1,
                "source": pdf_path,
                "text": text.strip()
            })

    document.close()

    return pages


def clean_text(text: str):
    # Remove excessive spaces and line breaks
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def create_chunks_from_pages(
    pages,
    chunk_size=800,
    chunk_overlap=150
):
    all_chunks = []

    for page_data in pages:

        page_number = page_data["page"]
        source = page_data["source"]

        text = clean_text(page_data["text"])

        if not text:
            continue

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                all_chunks.append({
                    "text": chunk_text,
                    "page": page_number,
                    "source": source
                })

            if end >= len(text):
                break

            start = end - chunk_overlap

    return all_chunks