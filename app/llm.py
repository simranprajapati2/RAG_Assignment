import httpx
import json
from app.config import settings


HEADERS = {
    "Authorization": f"Bearer {settings.API_KEY}",
    "Content-Type": "application/json"
}


CHAT_URL = (
    settings.NUGEN_BASE_URL.rstrip("/")
    + settings.CHAT_ENDPOINT
)


def generate_answer(question: str, context: str):

    system_prompt = """
You are a document question-answering assistant.

Answer the user's question using ONLY the provided
document context.

Do not invent information.

If the answer is not present in the context, say:

"I could not find the answer in the provided document."

Keep the answer clear and concise.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Answer using only the document context.
"""

    with httpx.Client(timeout=180.0) as client:

        response = client.post(
            CHAT_URL,
            headers=HEADERS,
            json={
                "model": settings.CHAT_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                        "name": "system"
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                        "name": "user"
                    }
                ],
                "max_tokens": 400,
                "prompt_truncate_len": 123,
                "temperature": 0.2,
                "stream": False
            }
        )

        response.raise_for_status()

        data = response.json()

        if "choices" in data:
            choice = data["choices"][0]

            if "message" in choice:
                return choice["message"]["content"].strip()

            if "text" in choice:
                return choice["text"].strip()

        if "text" in data:
            return data["text"].strip()

        raise ValueError(
            f"Unexpected Nugen response format: {data}"
        )


def stream_answer(question: str, context: str):

    system_prompt = """
You are a document question-answering assistant.

Answer the user's question using ONLY the provided
document context.

Do not invent information.

If the answer is not present in the context, say:

"I could not find the answer in the provided document."

Keep the answer clear and concise.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Answer using only the document context.
"""

    payload = {
        "model": settings.CHAT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
                "name": "system"
            },
            {
                "role": "user",
                "content": user_prompt,
                "name": "user"
            }
        ],
        "max_tokens": 400,
        "prompt_truncate_len": 123,
        "temperature": 0.2,
        "stream": True
    }

    with httpx.Client(timeout=180.0) as client:

        with client.stream(
            "POST",
            CHAT_URL,
            headers=HEADERS,
            json=payload
        ) as response:

            response.raise_for_status()

            for line in response.iter_lines():

                if not line:
                    continue

                # Nugen/OpenAI-style SSE response
                if line.startswith("data:"):
                    data = line[5:].strip()
                else:
                    data = line.strip()

                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if "choices" not in chunk:
                    continue

                choice = chunk["choices"][0]

                # Streaming chat format
                if "delta" in choice:

                    delta = choice["delta"]

                    content = delta.get(
                        "content",
                        ""
                    )

                    if content:
                        yield content

                # Fallback format
                elif "text" in choice:

                    text = choice["text"]

                    if text:
                        yield text