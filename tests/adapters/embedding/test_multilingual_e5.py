"""Tests de MultilingualE5Embedder avec les vrais poids intfloat/multilingual-e5-base."""

import math

from sentence_transformers import SentenceTransformer

from src.adapters.embedding.multilingual_e5 import MultilingualE5Embedder

_DIM = 768  # dimension native de multilingual-e5-base


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


class TestMultilingualE5EmbedderRealWeights:
    def test_embed_returns_correct_dimension_and_order(self, e5_model: SentenceTransformer) -> None:
        embedder = MultilingualE5Embedder(model=e5_model, batch_size=8)
        vectors = embedder.embed(["le xenon alimente les moteurs", "le menu du jour propose des pates"])

        assert len(vectors) == 2
        assert all(len(v) == _DIM for v in vectors)

    def test_embed_empty_list(self, e5_model: SentenceTransformer) -> None:
        embedder = MultilingualE5Embedder(model=e5_model, batch_size=8)
        assert embedder.embed([]) == []

    def test_embed_query_matches_embed_dimension(self, e5_model: SentenceTransformer) -> None:
        embedder = MultilingualE5Embedder(model=e5_model, batch_size=8)
        vector = embedder.embed_query("quel carburant utilise le systeme de propulsion ?")
        assert len(vector) == _DIM

    def test_semantically_related_texts_are_closer(self, e5_model: SentenceTransformer) -> None:
        embedder = MultilingualE5Embedder(model=e5_model, batch_size=8)

        query = embedder.embed_query("quel carburant utilise le systeme de propulsion ?")
        [related, unrelated] = embedder.embed(
            [
                "le systeme de propulsion utilise du xenon comme carburant",
                "le menu du refectoire propose des pates ce midi",
            ]
        )

        assert _cosine(query, related) > _cosine(query, unrelated)
