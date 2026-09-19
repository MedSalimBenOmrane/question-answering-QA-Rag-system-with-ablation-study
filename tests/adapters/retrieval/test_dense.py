"""Tests de DenseRetriever : une requete paraphrasee retrouve le bon chunk.

Vraie recherche semantique : vrai embedder BGE-M3, vrai VectorStore Chroma.
"""

from src.adapters.embedding.bge_m3 import BgeM3Embedder
from src.adapters.retrieval.dense import DenseRetriever
from src.adapters.vectorstore.chroma import ChromaVectorStore
from src.domain.models import Chunk


class TestDenseRetriever:
    def test_paraphrased_query_retrieves_relevant_chunk(
        self,
        mini_corpus: list[Chunk],
        chroma_store: ChromaVectorStore,
        bge_m3_embedder: BgeM3Embedder,
    ) -> None:
        chroma_store.add(mini_corpus, bge_m3_embedder.embed([c.text for c in mini_corpus]))
        retriever = DenseRetriever(embedder=bge_m3_embedder, vectorstore=chroma_store)

        # aucun mot en commun avec "xenon"/"fuel"/"thrusters", mais meme sens
        results = retriever.retrieve("What propellant powers the station's engines?", k=2)

        assert results[0].chunk.id == "c1"

    def test_returns_k_results(
        self,
        mini_corpus: list[Chunk],
        chroma_store: ChromaVectorStore,
        bge_m3_embedder: BgeM3Embedder,
    ) -> None:
        chroma_store.add(mini_corpus, bge_m3_embedder.embed([c.text for c in mini_corpus]))
        retriever = DenseRetriever(embedder=bge_m3_embedder, vectorstore=chroma_store)

        results = retriever.retrieve("emergency exits", k=3)

        assert len(results) == 3
