"""The RAG core: chunk the resume, embed it, and save or load the index."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

CHUNKS_FILE = "index.json"
VECTORS_FILE = "vectors.npy"
HEADING_SEPARATOR = " › "


class EmbeddingClient(Protocol):
    """Adapter around an embedding API, so tests can pass a fake."""

    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingModelMismatch(Exception):
    """The saved vectors were made by a different embedding model than the one configured."""


@dataclass(frozen=True)
class Chunk:
    id: int
    heading: str
    text: str  # the heading followed by the body: this is what gets embedded


@dataclass(frozen=True)
class Index:
    chunks: list[Chunk]
    vectors: np.ndarray  # one L2-normalised row per chunk
    embedding_model: str


def build_index(markdown: str, embedder: EmbeddingClient) -> Index:
    """Chunk the resume Markdown and embed every chunk."""
    chunks = chunk_resume(markdown)
    if not chunks:
        raise ValueError("The resume Markdown produced no chunks to embed.")
    vectors = np.array(embedder.embed([c.text for c in chunks]), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("The embedding client returned a zero vector, which can't be normalised.")
    vectors /= norms
    return Index(chunks=chunks, vectors=vectors, embedding_model=embedder.model)


def chunk_resume(markdown: str) -> list[Chunk]:
    """Split the resume into one chunk per `###` entry, or per `##` section without entries.

    The title (`#`) and its tagline form a chunk of their own.
    """
    sections: list[tuple[str, list[str]]] = []  # (heading, body lines)
    section = ""
    for line in markdown.splitlines():
        if line.startswith("### "):
            sections.append((f"{section}{HEADING_SEPARATOR}{line.removeprefix("### ").strip()}", []))
        elif line.startswith("## "):
            section = line.removeprefix("## ").strip()
            sections.append((section, []))
        elif line.startswith("# "):
            sections.append((line.removeprefix("# ").strip(), []))
        elif sections:
            sections[-1][1].append(line)

    chunks: list[Chunk] = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue  # a bare `##` that only introduces `###` entries
        chunks.append(Chunk(id=len(chunks), heading=heading, text=f"{heading}\n\n{body}"))
    return chunks


def save_index(index: Index, directory: Path) -> None:
    """Write chunks and the model name as JSON, and the vectors as a NumPy array."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "embedding_model": index.embedding_model,
        "chunks": [asdict(c) for c in index.chunks],
    }
    (directory / CHUNKS_FILE).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.save(directory / VECTORS_FILE, index.vectors)


def load_index(directory: Path, embedding_model: str) -> Index:
    """Load a saved index, refusing one built with a different embedding model."""
    payload = json.loads((directory / CHUNKS_FILE).read_text(encoding="utf-8"))
    if payload["embedding_model"] != embedding_model:
        raise EmbeddingModelMismatch(
            f"Index in {directory} was built with '{payload['embedding_model']}' but the configured "
            f"model is '{embedding_model}'. Rebuild it with `resume-assist build-index`."
        )
    chunks = [Chunk(**c) for c in payload["chunks"]]
    vectors = np.load(directory / VECTORS_FILE)
    if len(vectors) != len(chunks):
        raise ValueError(
            f"Index in {directory} has {len(chunks)} chunks but {len(vectors)} vector rows. "
            "Rebuild it with `resume-assist build-index`."
        )
    return Index(chunks=chunks, vectors=vectors, embedding_model=payload["embedding_model"])
