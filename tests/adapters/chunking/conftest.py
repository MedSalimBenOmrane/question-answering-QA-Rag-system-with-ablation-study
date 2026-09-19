"""Fixture partagee : vrai TokenCounter (tiktoken cl100k_base), pas de mock."""

import tiktoken
import pytest

from src.adapters.llm.token_counter import TiktokenTokenCounter


@pytest.fixture(scope="session")
def real_token_counter() -> TiktokenTokenCounter:
    return TiktokenTokenCounter(tiktoken.get_encoding("cl100k_base"))
