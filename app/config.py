from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    API_KEY: str

    NUGEN_BASE_URL: str = "https://api.nugen.in/api/v3/inference"

    EMBED_ENDPOINT: str = "/embeddings"
    CHAT_ENDPOINT: str = "/chat/completions"

    EMBED_MODEL: str = "model_01m39qzr2s25e4nv"
    CHAT_MODEL: str = "model_01m39qzr2s25e4nv"

    EMBEDDING_DIM: int = 768

    QDRANT_PATH: str = "./data/qdrant"
    QDRANT_COLLECTION: str = "pdf_documents"

    TOP_K: int = 10
    RERANK_TOP_K: int = 3

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 80

    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()