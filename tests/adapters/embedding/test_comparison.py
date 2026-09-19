"""Comparaison des deux strategies d'embedding sur un meme mini-corpus.

Verifie que BGE-M3 et multilingual-e5, bien qu'entraines et utilises
differemment (e5 exige des prefixes query/passage, pas BGE-M3), retrouvent
tous les deux le bon chunk pertinent une fois indexes dans Chroma. C'est ce
qui rend les deux strategies reellement interchangeables et comparables
(cf. etude d'ablation, CLAUDE.md).
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.adapters.embedding.bge_m3 import BgeM3Embedder
from src.adapters.embedding.multilingual_e5 import MultilingualE5Embedder
from src.adapters.vectorstore.chroma import ChromaVectorStore
from src.domain.models import Chunk
from src.domain.ports import Embedder

_CHUNKS = [
    Chunk(id="c1", text="Le systeme de propulsion utilise du xenon comme carburant.", source="file01", n_tokens=9),
    Chunk(id="c2", text="Le menu du refectoire propose des pates ce midi.", source="file02", n_tokens=8),
    Chunk(id="c3", text="Les procedures de securite listent les issues de secours.", source="file03", n_tokens=8),
]
_QUERY = "Quel carburant alimente les moteurs de la station ?"


def _index_and_search(embedder: Embedder, tmp_path: Path, name: str) -> str:
    client = chromadb.PersistentClient(path=str(tmp_path / name))
    store = ChromaVectorStore(client=client, collection_name=name)
    store.add(_CHUNKS, embedder.embed([c.text for c in _CHUNKS]))
    results = store.search(embedder.embed_query(_QUERY), k=1)
    return results[0].chunk.id


class TestEmbedderComparison:
    def test_both_strategies_retrieve_the_same_relevant_chunk(
        self, tmp_path: Path, bge_m3_model: SentenceTransformer, e5_model: SentenceTransformer
    ) -> None:
        bge_m3_top = _index_and_search(BgeM3Embedder(model=bge_m3_model, batch_size=8), tmp_path, "bge_m3")
        e5_top = _index_and_search(
            MultilingualE5Embedder(model=e5_model, batch_size=8), tmp_path, "multilingual_e5"
        )

        assert bge_m3_top == "c1"
        assert e5_top == "c1"
