import hashlib

import numpy as np
import pytest

from resume_assist.rag import EmbeddingModelMismatch, build_index, load_index, save_index

RESUME = """# Jane Doe

Engineer | Consultant

## Experience

### Acme Corp, Senior Engineer, 2022 – 2024

Employer: Acme Corp
Dates: 2022 – 2024

- Cut costs by 30%.

### Beta Ltd, Engineer, 2019 – 2022

- Built a payments API.

## Skills

- Python, SQL

## Education

### State University, BSc Physics

Grade: First
"""


class FakeEmbedder:
    """Deterministic, offline stand-in for the OpenAI embedding client."""

    def __init__(self, model: str = "fake-embed-1") -> None:
        self.model = model
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [_vector(text) for text in texts]


def _vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [float(b) + 1.0 for b in digest[:8]]


def test_each_role_is_one_chunk_with_its_heading_prepended():
    index = build_index(RESUME, FakeEmbedder())

    acme = next(c for c in index.chunks if "Acme Corp" in c.heading)
    assert acme.heading == "Experience › Acme Corp, Senior Engineer, 2022 – 2024"
    assert acme.text.startswith(acme.heading)
    assert "Dates: 2022 – 2024" in acme.text
    assert "Cut costs by 30%." in acme.text
    assert "Built a payments API." not in acme.text


def test_skills_block_and_education_entry_are_chunks():
    index = build_index(RESUME, FakeEmbedder())

    headings = [c.heading for c in index.chunks]
    assert "Skills" in headings
    assert "Education › State University, BSc Physics" in headings
    skills = next(c for c in index.chunks if c.heading == "Skills")
    assert "Python, SQL" in skills.text


def test_chunks_have_stable_sequential_ids():
    index = build_index(RESUME, FakeEmbedder())

    assert [c.id for c in index.chunks] == list(range(len(index.chunks)))


def test_vectors_are_normalised_with_one_row_per_chunk():
    index = build_index(RESUME, FakeEmbedder())

    assert index.vectors.shape[0] == len(index.chunks)
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0)


def test_chunk_text_including_heading_is_what_gets_embedded():
    embedder = FakeEmbedder()
    index = build_index(RESUME, embedder)

    assert embedder.calls == [[c.text for c in index.chunks]]


def test_saved_index_round_trips(tmp_path):
    index = build_index(RESUME, FakeEmbedder())

    save_index(index, tmp_path / "idx")
    loaded = load_index(tmp_path / "idx", embedding_model="fake-embed-1")

    assert loaded.chunks == index.chunks
    assert loaded.embedding_model == "fake-embed-1"
    assert np.allclose(loaded.vectors, index.vectors)


def test_loading_an_index_built_with_a_different_model_fails_clearly(tmp_path):
    save_index(build_index(RESUME, FakeEmbedder("old-model")), tmp_path / "idx")

    with pytest.raises(EmbeddingModelMismatch, match="old-model.*new-model"):
        load_index(tmp_path / "idx", embedding_model="new-model")


def test_building_from_a_resume_with_no_content_fails_clearly():
    with pytest.raises(ValueError, match="no chunks"):
        build_index("", FakeEmbedder())


def test_a_zero_vector_from_the_embedder_fails_clearly():
    class ZeroEmbedder(FakeEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0, 0.0] for _ in texts]

    with pytest.raises(ValueError, match="zero vector"):
        build_index(RESUME, ZeroEmbedder())


def test_loading_vectors_that_do_not_match_the_chunks_fails_clearly(tmp_path):
    index = build_index(RESUME, FakeEmbedder())
    save_index(index, tmp_path / "idx")
    np.save(tmp_path / "idx" / "vectors.npy", index.vectors[:-1])

    with pytest.raises(ValueError, match="rows"):
        load_index(tmp_path / "idx", embedding_model="fake-embed-1")
