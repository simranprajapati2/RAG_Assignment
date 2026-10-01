from pydantic import BaseModel, Field


class AskRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask about the PDF"
    )


class RetrievedChunk(BaseModel):

    rank: int
    page: int
    source: str
    score: float
    text: str


class AskResponse(BaseModel):

    question: str
    answer: str

    retrieved_chunks: list[RetrievedChunk]

    reranked_chunks: list[RetrievedChunk]