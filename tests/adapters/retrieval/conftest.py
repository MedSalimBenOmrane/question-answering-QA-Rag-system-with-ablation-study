"""Fixtures partagees pour les tests de retrieval : mini-corpus et Chroma reel."""

from pathlib import Path

import chromadb
import pytest
import tiktoken

from src.adapters.llm.token_counter import TiktokenTokenCounter
from src.adapters.vectorstore.chroma import ChromaVectorStore
from src.domain.models import Chunk

_TEXTS = {
    "c1": "The propulsion system uses xenon as fuel for the ion thrusters.",
    "c2": "The cafeteria menu today offers pasta and a green salad.",
    "c3": "Safety procedures list all emergency exits located on deck two.",
    "c4": "The boot sequence initializes core services before the AI layer.",
}


@pytest.fixture(scope="session")
def mini_corpus() -> list[Chunk]:
    counter = TiktokenTokenCounter(tiktoken.get_encoding("cl100k_base"))
    return [
        Chunk(id=chunk_id, text=text, source=f"{chunk_id}.md", n_tokens=counter.count(text))
        for chunk_id, text in _TEXTS.items()
    ]


@pytest.fixture
def chroma_store(tmp_path: Path) -> ChromaVectorStore:
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    return ChromaVectorStore(client=client, collection_name="retrieval_test")
