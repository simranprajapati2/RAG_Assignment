from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

from app.config import settings


client = QdrantClient(
    path=settings.QDRANT_PATH
)


def create_collection():

    collections = client.get_collections()

    existing_names = [
        collection.name
        for collection in collections.collections
    ]

    if settings.QDRANT_COLLECTION not in existing_names:

        print(
            "\nCreating Qdrant collection..."
        )

        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,

            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )

        print(
            "Collection created:",
            settings.QDRANT_COLLECTION
        )


def clear_collection():

    collections = client.get_collections()

    existing_names = [
        collection.name
        for collection in collections.collections
    ]

    if settings.QDRANT_COLLECTION in existing_names:

        client.delete_collection(
            collection_name=settings.QDRANT_COLLECTION
        )

    create_collection()


def store_chunks(
    embeddings,
    chunks
):

    points = []

    for index, (embedding, chunk) in enumerate(
        zip(embeddings, chunks)
    ):

        points.append(
            PointStruct(
                id=index,

                vector=embedding,

                payload={
                    "text": chunk["text"],
                    "page": chunk["page"],
                    "source": chunk["source"]
                }
            )
        )

    client.upsert(
        collection_name=settings.QDRANT_COLLECTION,
        points=points
    )

    return len(points)


def search(
    query_vector,
    limit=None
):

    if limit is None:
        limit = settings.TOP_K

    results = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True
    ).points

    output = []

    for result in results:

        payload = result.payload

        output.append({
            "page": payload["page"],
            "source": payload["source"],
            "text": payload["text"],
            "score": float(result.score)
        })

    return output