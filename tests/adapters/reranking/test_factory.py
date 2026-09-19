"""Tests de la factory de reranking (active/desactive via config, erreurs)."""

import pytest

from src.adapters.reranking.cross_encoder import CrossEncoderReranker
from src.adapters.reranking.factory import create_reranker
from src.adapters.reranking.noop import NoOpReranker


class TestCreateReranker:
    def test_disabled_returns_noop(self) -> None:
        reranker = create_reranker({"enabled": False})
        assert isinstance(reranker, NoOpReranker)

    def test_enabled_returns_real_cross_encoder(self) -> None:
        reranker = create_reranker(
            {
                "enabled": True,
                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "device": "cpu",
            }
        )
        assert isinstance(reranker, CrossEncoderReranker)

    def test_missing_enabled_key_raises(self) -> None:
        with pytest.raises(KeyError):
            create_reranker({})

    def test_missing_model_name_when_enabled_raises(self) -> None:
        with pytest.raises(KeyError):
            create_reranker({"enabled": True})
