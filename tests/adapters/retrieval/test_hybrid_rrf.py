"""Tests de HybridRRFRetriever : fusion RRF sur une requete lexicale ET paraphrasee.

C'est l'interet de l'hybride : BM25 seul rate une requete paraphrasee sans
recouvrement lexical, un dense seul est moins fiable sur une requete-citation
exacte. La fusion doit retrouver le bon chunk dans les deux cas.
"""

from src.adapters.embedding.bge_m3 import BgeM3Embedder
from src.adapters.retrieval.bm25 import BM25Retriever
from src.adapters.retrieval.dense import DenseRetriever
from src.adapters.retrieval.hybrid_rrf import HybridRRFRetriever
from src.adapters.vectorstore.chroma import ChromaVectorStore
from src.domain.models import Chunk


def _build_hybrid(
    mini_corpus: list[Chunk], chroma_store: ChromaVectorStore, bge_m3_embedder: BgeM3Embedder
) -> HybridRRFRetriever:
    chroma_store.add(mini_corpus, bge_m3_embedder.embed([c.text for c in mini_corpus]))
    dense = DenseRetriever(embedder=bge_m3_embedder, vectorstore=chroma_store)
    sparse = BM25Retriever(chunks=mini_corpus, k1=1.5, b=0.75)
    return HybridRRFRetriever(dense=dense, sparse=sparse, rrf_k=60)


class TestHybridRRFRetriever:
    def test_exact_lexical_query_retrieves_relevant_chunk(
        self,
        mini_corpus: list[Chunk],
        chroma_store: ChromaVectorStore,
        bge_m3_embedder: BgeM3Embedder,
    ) -> None:
        retriever = _build_hybrid(mini_corpus, chroma_store, bge_m3_embedder)

        results = retriever.retrieve("xenon fuel for the ion thrusters", k=2)

        assert results[0].chunk.id == "c1"

    def test_paraphrased_query_retrieves_relevant_chunk(
        self,
        mini_corpus: list[Chunk],
        chroma_store: ChromaVectorStore,
        bge_m3_embedder: BgeM3Embedder,
    ) -> None:
        retriever = _build_hybrid(mini_corpus, chroma_store, bge_m3_embedder)

        results = retriever.retrieve("What propellant powers the station's engines?", k=2)

        assert results[0].chunk.id == "c1"

    def test_fused_scores_are_positive_and_sorted_descending(
        self,
        mini_corpus: list[Chunk],
        chroma_store: ChromaVectorStore,
        bge_m3_embedder: BgeM3Embedder,
    ) -> None:
        retriever = _build_hybrid(mini_corpus, chroma_store, bge_m3_embedder)

        results = retriever.retrieve("xenon fuel for the ion thrusters", k=4)
        scores = [r.score for r in results]

        assert all(s > 0 for s in scores)
        assert scores == sorted(scores, reverse=True)
