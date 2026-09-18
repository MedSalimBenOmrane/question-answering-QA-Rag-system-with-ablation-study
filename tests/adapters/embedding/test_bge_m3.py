"""Tests de l'Embedder BGE-M3 (wrapper sentence-transformers + factory).

Utilise un modele factice (pas de telechargement des poids BGE-M3, ~2 Go)
pour tester la logique de l'adapter : ordre preserve, conversion en
list[float], parametres (batch_size, device) correctement transmis.
"""

from typing import Any

import numpy as np
import pytest

from src.adapters.embedding.bge_m3 import BgeM3Embedder, create_embedder


class FakeSentenceTransformer:
    """Modele factice : encode chaque texte en un vecteur base sur sa longueur."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device

    def encode(self, texts: list[str], batch_size: int, convert_to_numpy: bool) -> Any:
        return np.array([[float(len(t)), float(len(t.split()))] for t in texts])


class TestBgeM3Embedder:
    def test_embed_preserves_order_and_returns_float_lists(self) -> None:
        embedder = BgeM3Embedder(model=FakeSentenceTransformer("BAAI/bge-m3"), batch_size=8)
        vectors = embedder.embed(["ab", "abcd efgh"])
        assert vectors == [[2.0, 1.0], [9.0, 2.0]]
        assert all(isinstance(v, list) for v in vectors)

    def test_embed_empty_list(self) -> None:
        embedder = BgeM3Embedder(model=FakeSentenceTransformer("BAAI/bge-m3"), batch_size=8)
        assert embedder.embed([]) == []

    def test_embed_query_returns_single_vector(self) -> None:
        embedder = BgeM3Embedder(model=FakeSentenceTransformer("BAAI/bge-m3"), batch_size=8)
        vector = embedder.embed_query("hello world")
        assert vector == [11.0, 2.0]


class TestCreateEmbedder:
    def test_wires_config_into_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake_ctor(model_name: str, device: str | None = None) -> FakeSentenceTransformer:
            captured["model_name"] = model_name
            captured["device"] = device
            return FakeSentenceTransformer(model_name, device)

        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer", fake_ctor, raising=False
        )

        embedder = create_embedder({"model_name": "BAAI/bge-m3", "device": "cpu", "batch_size": 4})

        assert isinstance(embedder, BgeM3Embedder)
        assert captured == {"model_name": "BAAI/bge-m3", "device": "cpu"}
        assert embedder.embed(["hi"]) == [[2.0, 1.0]]

    def test_missing_model_name_raises(self) -> None:
        with pytest.raises(KeyError):
            create_embedder({})
