"""Tests de la factory de retrieval (dispatch dense / bm25 / hybride, erreurs)."""

import pytest

from src.adapters.embedding.bge_m3 import BgeM3Embedder
from src.adapters.retrieval.bm25 import BM25Retriever
from src.adapters.retrieval.dense import DenseRetriever
from src.adapters.retrieval.factory import create_retriever
from src.adapters.retrieval.hybrid_rrf import HybridRRFRetriever
from src.adapters.vectorstore.chroma import ChromaVectorStore
from src.domain.models import Chunk


class TestCreateRetriever:
    def test_creates_dense(
        self, mini_corpus: list[Chunk], chroma_store: ChromaVectorStore, bge_m3_embedder: BgeM3Embedder
    ) -> None:
        retriever = create_retriever(
            {"hybrid": False, "strategy": "dense"}, bge_m3_embedder, chroma_store, mini_corpus
        )
        assert isinstance(retriever, DenseRetriever)

    def test_creates_bm25(
        self, mini_corpus: list[Chunk], chroma_store: ChromaVectorStore, bge_m3_embedder: BgeM3Embedder
    ) -> None:
        retriever = create_retriever(
            {"hybrid": False, "strategy": "bm25", "bm25": {"k1": 1.5, "b": 0.75}},
            bge_m3_embedder,
            chroma_store,
            mini_corpus,
        )
        assert isinstance(retriever, BM25Retriever)

    def test_creates_hybrid(
        self, mini_corpus: list[Chunk], chroma_store: ChromaVectorStore, bge_m3_embedder: BgeM3Embedder
    ) -> None:
        retriever = create_retriever(
            {"hybrid": True, "bm25": {"k1": 1.5, "b": 0.75}, "rrf_k": 60},
            bge_m3_embedder,
            chroma_store,
            mini_corpus,
        )
        assert isinstance(retriever, HybridRRFRetriever)

    def test_missing_hybrid_key_raises(
        self, mini_corpus: list[Chunk], chroma_store: ChromaVectorStore, bge_m3_embedder: BgeM3Embedder
    ) -> None:
        with pytest.raises(KeyError):
            create_retriever({}, bge_m3_embedder, chroma_store, mini_corpus)

    def test_missing_strategy_when_not_hybrid_raises(
        self, mini_corpus: list[Chunk], chroma_store: ChromaVectorStore, bge_m3_embedder: BgeM3Embedder
    ) -> None:
        with pytest.raises(KeyError):
            create_retriever({"hybrid": False}, bge_m3_embedder, chroma_store, mini_corpus)

    def test_unknown_strategy_raises(
        self, mini_corpus: list[Chunk], chroma_store: ChromaVectorStore, bge_m3_embedder: BgeM3Embedder
    ) -> None:
        with pytest.raises(ValueError):
            create_retriever(
                {"hybrid": False, "strategy": "nope"}, bge_m3_embedder, chroma_store, mini_corpus
            )

    def test_missing_rrf_k_when_hybrid_raises(
        self, mini_corpus: list[Chunk], chroma_store: ChromaVectorStore, bge_m3_embedder: BgeM3Embedder
    ) -> None:
        with pytest.raises(KeyError):
            create_retriever(
                {"hybrid": True, "bm25": {"k1": 1.5, "b": 0.75}},
                bge_m3_embedder,
                chroma_store,
                mini_corpus,
            )

    def test_missing_bm25_params_raises(
        self, mini_corpus: list[Chunk], chroma_store: ChromaVectorStore, bge_m3_embedder: BgeM3Embedder
    ) -> None:
        with pytest.raises(KeyError):
            create_retriever(
                {"hybrid": False, "strategy": "bm25"}, bge_m3_embedder, chroma_store, mini_corpus
            )
