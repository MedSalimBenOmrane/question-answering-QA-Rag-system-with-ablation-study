"""Fixture partagee : vrai modele cross-encoder, charge une seule fois."""

import pytest
from sentence_transformers import CrossEncoder


@pytest.fixture(scope="session")
def cross_encoder_model() -> CrossEncoder:
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
