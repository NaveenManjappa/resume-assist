"""Adapters that put the OpenAI SDK behind the RAG core's client interfaces."""

from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"


class OpenAIEmbedder:
    def __init__(self, client: OpenAI | None = None, model: str = EMBEDDING_MODEL) -> None:
        self.client = client or OpenAI()
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
