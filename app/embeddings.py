import time
import httpx

from app.config import settings


HEADERS = {
    "Authorization": f"Bearer {settings.API_KEY}",
    "Content-Type": "application/json"
}


EMBED_URL = (
    settings.NUGEN_BASE_URL.rstrip("/")
    + settings.EMBED_ENDPOINT
)


def get_embedding(
    text: str,
    max_retries: int = 5
):

    print("\n================================")
    print("NUGEN EMBEDDING")
    print("URL:", EMBED_URL)
    print("MODEL:", settings.EMBED_MODEL)
    print("================================")

    last_error = None

    for attempt in range(1, max_retries + 1):

        try:

            print(
                f"Attempt {attempt}/{max_retries}"
            )

            with httpx.Client(
                timeout=120.0
            ) as client:

                response = client.post(
                    EMBED_URL,
                    headers=HEADERS,
                    json={
                        "model": settings.EMBED_MODEL,
                        "input": text
                    }
                )

            print(
                "Embedding status:",
                response.status_code
            )

            # Success
            if response.status_code == 200:

                data = response.json()

                embedding = data["data"][0]["embedding"]

                print(
                    "Embedding dimension:",
                    len(embedding)
                )

                return embedding

            # Temporary server errors
            if response.status_code in [
                429,
                500,
                502,
                503,
                504
            ]:

                print(
                    f"Temporary Nugen error "
                    f"{response.status_code}"
                )

                print(
                    "Retrying..."
                )

                last_error = response.text

                time.sleep(
                    2 ** (attempt - 1)
                )

                continue

            # Other errors
            print(
                "Embedding error:",
                response.text
            )

            response.raise_for_status()

        except (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.NetworkError
        ) as error:

            print(
                "Network/timeout error:",
                error
            )

            last_error = str(error)

            time.sleep(
                2 ** (attempt - 1)
            )

    raise RuntimeError(
        "Nugen embedding failed after "
        f"{max_retries} attempts.\n"
        f"Last error: {last_error}"
    )