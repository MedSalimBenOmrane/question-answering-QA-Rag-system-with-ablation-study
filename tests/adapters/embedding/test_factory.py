"""Tests de la factory d'embedding (dispatch reel bge_m3 / multilingual_e5)."""

import pytest

from src.adapters.embedding.bge_m3 import BgeM3Embedder
from src.adapters.embedding.factory import create_embedder
from src.adapters.embedding.multilingual_e5 import MultilingualE5Embedder


class TestCreateEmbedder:
    def test_creates_bge_m3(self) -> None:
        embedder = create_embedder(
            {"provider": "bge_m3", "bge_m3": {"model_name": "BAAI/bge-m3", "device": "cpu"}}
        )
        assert isinstance(embedder, BgeM3Embedder)
        assert len(embedder.embed_query("test")) == 1024

    def test_creates_multilingual_e5(self) -> None:
        embedder = create_embedder(
            {
                "provider": "multilingual_e5",
                "multilingual_e5": {"model_name": "intfloat/multilingual-e5-base", "device": "cpu"},
            }
        )
        assert isinstance(embedder, MultilingualE5Embedder)
        assert len(embedder.embed_query("test")) == 768

    def test_missing_provider_raises(self) -> None:
        with pytest.raises(KeyError):
            create_embedder({})

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError):
            create_embedder({"provider": "does-not-exist"})

    def test_missing_model_name_raises(self) -> None:
        with pytest.raises(KeyError):
            create_embedder({"provider": "bge_m3", "bge_m3": {}})
