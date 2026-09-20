"""Tests du VectorStore Chroma : add puis search sur un mini-corpus factice.

Utilise des vecteurs fabriques a la main (pas le vrai embedder BGE-M3) pour
que la pertinence attendue soit connue d'avance et le test deterministe.
"""

from pathlib import Path

import chromadb
import pytest

from src.adapters.vectorstore.chroma import ChromaVectorStore, create_vectorstore
from src.domain.models import Chunk

_CHUNKS = [
    Chunk(id="c1", text="Le systeme de propulsion utilise du xenon.", source="file01", n_tokens=8),
    Chunk(id="c2", text="Le menu du refectoire propose des pates.", source="file02", n_tokens=7),
    Chunk(id="c3", text="Les procedures de securite listent les issues.", source="file03", n_tokens=7),
]

# vecteurs 4D fabriques a la main : c1 est clairement le plus proche de la requete
_VECTORS = [
    [1.0, 0.0, 0.0, 0.0],  # c1
    [0.0, 1.0, 0.0, 0.0],  # c2
    [0.0, 0.0, 1.0, 0.0],  # c3
]
_QUERY_VECTOR = [0.95, 0.05, 0.0, 0.0]  # proche de c1


def _make_store(tmp_path: Path, collection_name: str = "test") -> ChromaVectorStore:
    client = chromadb.PersistentClient(path=str(tmp_path))
    return ChromaVectorStore(client=client, collection_name=collection_name)


class TestChromaVectorStore:
    def test_add_then_search_returns_relevant_scored_chunks(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.add(_CHUNKS, _VECTORS)

        results = store.search(_QUERY_VECTOR, k=2)

        assert len(results) == 2
        assert results[0].chunk.id == "c1"
        assert results[0].chunk.source == "file01"
        assert results[0].score > results[1].score

    def test_search_preserves_chunk_fields(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.add(_CHUNKS, _VECTORS)

        results = store.search(_QUERY_VECTOR, k=1)

        top = results[0].chunk
        assert top.text == "Le systeme de propulsion utilise du xenon."
        assert top.n_tokens == 8

    def test_persists_across_client_instances(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, collection_name="persistent")
        store.add(_CHUNKS, _VECTORS)

        reopened = _make_store(tmp_path, collection_name="persistent")
        results = reopened.search(_QUERY_VECTOR, k=1)

        assert results[0].chunk.id == "c1"

    def test_add_rejects_mismatched_lengths(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        with pytest.raises(ValueError):
            store.add(_CHUNKS, _VECTORS[:1])

    def test_add_empty_is_noop(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.add([], [])
        assert store.search(_QUERY_VECTOR, k=5) == []

    def test_search_includes_embedding_in_metadata(self, tmp_path: Path) -> None:
        """Metadonnee conservee pour un usage eventuel par une strategie de
        selection (cf. domain/budget.py) - pas de comportement propre a une
        strategie particuliere, ce test verifie seulement sa presence."""
        store = _make_store(tmp_path)
        store.add(_CHUNKS, _VECTORS)

        results = store.search(_QUERY_VECTOR, k=3)

        by_id = {r.chunk.id: r.chunk for r in results}
        assert by_id["c1"].metadata["embedding"] == pytest.approx([1.0, 0.0, 0.0, 0.0])
        assert by_id["c2"].metadata["embedding"] == pytest.approx([0.0, 1.0, 0.0, 0.0])


class TestCreateVectorstore:
    def test_creates_persistent_store_from_config(self, tmp_path: Path) -> None:
        store = create_vectorstore(
            {"persist_directory": str(tmp_path / "chroma"), "collection_name": "from_config"}
        )
        assert isinstance(store, ChromaVectorStore)

        store.add(_CHUNKS, _VECTORS)
        results = store.search(_QUERY_VECTOR, k=1)
        assert results[0].chunk.id == "c1"

    def test_missing_keys_raise(self) -> None:
        with pytest.raises(KeyError):
            create_vectorstore({})
