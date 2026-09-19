"""Tests de BM25Retriever : une requete lexicale exacte retrouve le bon chunk."""

from src.adapters.retrieval.bm25 import BM25Retriever
from src.domain.models import Chunk


class TestBM25Retriever:
    def test_exact_lexical_query_retrieves_relevant_chunk(
        self, mini_corpus: list[Chunk]
    ) -> None:
        retriever = BM25Retriever(chunks=mini_corpus, k1=1.5, b=0.75)

        results = retriever.retrieve("xenon fuel for the ion thrusters", k=2)

        assert results[0].chunk.id == "c1"

    def test_no_lexical_overlap_scores_zero(self, mini_corpus: list[Chunk]) -> None:
        retriever = BM25Retriever(chunks=mini_corpus, k1=1.5, b=0.75)

        results = retriever.retrieve("zzzznotinthecorpus", k=4)

        assert all(r.score == 0.0 for r in results)

    def test_empty_corpus_returns_empty(self) -> None:
        retriever = BM25Retriever(chunks=[], k1=1.5, b=0.75)
        assert retriever.retrieve("anything", k=5) == []
