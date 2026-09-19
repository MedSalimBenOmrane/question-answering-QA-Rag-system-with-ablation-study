"""Tests de NoOpReranker : reranking desactive, ordre du retrieval preserve."""

from src.adapters.reranking.noop import NoOpReranker
from src.domain.models import Chunk, ScoredChunk

_CHUNKS = [
    ScoredChunk(chunk=Chunk(id="c2", text="deuxieme", source="c2.md", n_tokens=1), score=0.9),
    ScoredChunk(chunk=Chunk(id="c3", text="troisieme", source="c3.md", n_tokens=1), score=0.8),
    ScoredChunk(chunk=Chunk(id="c1", text="premier", source="c1.md", n_tokens=1), score=0.1),
]


class TestNoOpReranker:
    def test_preserves_original_order(self) -> None:
        reranker = NoOpReranker()
        result = reranker.rerank("peu importe", _CHUNKS, top_n=3)
        assert [sc.chunk.id for sc in result] == [sc.chunk.id for sc in _CHUNKS]

    def test_respects_top_n(self) -> None:
        reranker = NoOpReranker()
        result = reranker.rerank("peu importe", _CHUNKS, top_n=2)
        assert [sc.chunk.id for sc in result] == ["c2", "c3"]
