"""Fixtures partagees : charge chaque modele reel une seule fois par session."""

import pytest
from sentence_transformers import SentenceTransformer


@pytest.fixture(scope="session")
def bge_m3_model() -> SentenceTransformer:
    return SentenceTransformer("BAAI/bge-m3", device="cpu")


@pytest.fixture(scope="session")
def e5_model() -> SentenceTransformer:
    return SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")
