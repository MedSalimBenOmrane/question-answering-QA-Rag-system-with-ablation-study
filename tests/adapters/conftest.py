"""Fixtures partagees a tous les tests d'adapters : vrais modeles, charges
une seule fois par session (jamais de mock)."""

import pytest
from sentence_transformers import SentenceTransformer

from src.adapters.embedding.bge_m3 import BgeM3Embedder
from src.adapters.embedding.multilingual_e5 import MultilingualE5Embedder


@pytest.fixture(scope="session")
def bge_m3_model() -> SentenceTransformer:
    return SentenceTransformer("BAAI/bge-m3", device="cpu")


@pytest.fixture(scope="session")
def e5_model() -> SentenceTransformer:
    return SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")


@pytest.fixture(scope="session")
def bge_m3_embedder(bge_m3_model: SentenceTransformer) -> BgeM3Embedder:
    return BgeM3Embedder(model=bge_m3_model, batch_size=8)


@pytest.fixture(scope="session")
def e5_embedder(e5_model: SentenceTransformer) -> MultilingualE5Embedder:
    return MultilingualE5Embedder(model=e5_model, batch_size=8)
