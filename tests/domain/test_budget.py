"""Tests de src/domain/budget.py (port Selector) : TopKSelector, create_selector.

Coeur du projet : ces tests verifient rigoureusement le respect du budget de
tokens (jamais depasse).

Note (chantier "relative_threshold", etape 1) : les tests des strategies par
densite avec anti-redondance (retirees de budget.py) ont ete supprimes de ce
fichier - liste exhaustive dans le rapport d'etape.
"""

import pytest

from src.domain.budget import TopKSelector, create_selector
from src.domain.models import Chunk, ScoredChunk
from src.domain.selection import RelativeThresholdSelector


def _chunk(chunk_id: str, n_tokens: int) -> Chunk:
    return Chunk(id=chunk_id, text=f"text of {chunk_id}", source="test.md", n_tokens=n_tokens)


def _total_tokens(chunks: list[Chunk]) -> int:
    return sum(c.n_tokens for c in chunks)


class TestBudgetNeverExceeded:
    """Invariant central : la somme des n_tokens selectionnes ne depasse jamais le budget."""

    CHUNKS = [
        ScoredChunk(chunk=_chunk("a", 350), score=0.9),
        ScoredChunk(chunk=_chunk("b", 150), score=0.8),
        ScoredChunk(chunk=_chunk("c", 150), score=0.75),
        ScoredChunk(chunk=_chunk("d", 140), score=0.7),
    ]
    BUDGET = 600

    def test_topk_never_exceeds_budget(self) -> None:
        result = TopKSelector().select("q", self.CHUNKS, self.BUDGET)
        assert _total_tokens(result) <= self.BUDGET


class TestBudgetExactlyReached:
    """Le budget peut etre atteint exactement (somme == budget), pas seulement approche."""

    def test_topk_includes_chunk_that_exactly_fills_remaining_budget(self) -> None:
        chunks = [
            ScoredChunk(chunk=_chunk("a", 300), score=1.0),
            ScoredChunk(chunk=_chunk("b", 300), score=0.9),
        ]
        result = TopKSelector().select("q", chunks, context_budget=600)

        assert {c.id for c in result} == {"a", "b"}
        assert _total_tokens(result) == 600


class TestBudgetZero:
    def test_topk_budget_zero_returns_empty(self) -> None:
        chunks = [ScoredChunk(chunk=_chunk("a", 10), score=0.9)]
        assert TopKSelector().select("q", chunks, context_budget=0) == []


class TestSingleChunkTooBig:
    def test_topk_single_oversized_chunk_returns_empty(self) -> None:
        chunks = [ScoredChunk(chunk=_chunk("a", 1000), score=0.9)]
        assert TopKSelector().select("q", chunks, context_budget=600) == []


class TestTopKStopsAtFirstChunkThatDoesNotFit:
    """TopKSelector s'arrete au premier chunk qui ne rentre pas (pas de packing)."""

    CHUNKS = [
        ScoredChunk(chunk=_chunk("A", 350), score=0.9),
        ScoredChunk(chunk=_chunk("B", 150), score=0.8),
        ScoredChunk(chunk=_chunk("C", 150), score=0.75),
        ScoredChunk(chunk=_chunk("D", 140), score=0.7),
    ]
    BUDGET = 600

    def test_topk_picks_a_and_b_then_stops(self) -> None:
        result = TopKSelector().select("q", self.CHUNKS, self.BUDGET)
        assert {c.id for c in result} == {"A", "B"}


class TestAntiLostInMiddleOrdering:
    def test_best_first_second_best_last(self) -> None:
        chunks = [
            ScoredChunk(chunk=_chunk("1st", 100), score=1.0),
            ScoredChunk(chunk=_chunk("2nd", 100), score=0.9),
            ScoredChunk(chunk=_chunk("3rd", 100), score=0.8),
            ScoredChunk(chunk=_chunk("4th", 100), score=0.7),
        ]
        result = TopKSelector().select("q", chunks, context_budget=400)
        assert [c.id for c in result] == ["1st", "3rd", "4th", "2nd"]

    def test_two_chunks_order_unchanged(self) -> None:
        chunks = [
            ScoredChunk(chunk=_chunk("1st", 100), score=1.0),
            ScoredChunk(chunk=_chunk("2nd", 100), score=0.9),
        ]
        result = TopKSelector().select("q", chunks, context_budget=200)
        assert [c.id for c in result] == ["1st", "2nd"]


_RELATIVE_THRESHOLD_CONFIG = {
    "min_abs_score": 1e-3,
    "log_margin": 1.5,
    "drop_ratio": 0.20,
    "max_chunks": 5,
    "min_chunks": 1,
}


class TestCreateSelector:
    def test_creates_topk(self) -> None:
        assert isinstance(create_selector({"strategy": "topk"}), TopKSelector)

    def test_missing_strategy_raises(self) -> None:
        with pytest.raises(KeyError):
            create_selector({})

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError):
            create_selector({"strategy": "does-not-exist"})

    def test_creates_relative_threshold(self) -> None:
        selector = create_selector(
            {"strategy": "relative_threshold", "relative_threshold": _RELATIVE_THRESHOLD_CONFIG}
        )
        assert isinstance(selector, RelativeThresholdSelector)

    def test_relative_threshold_missing_section_raises(self) -> None:
        with pytest.raises(KeyError):
            create_selector({"strategy": "relative_threshold"})

    def test_relative_threshold_missing_key_raises(self) -> None:
        incomplete = dict(_RELATIVE_THRESHOLD_CONFIG)
        del incomplete["max_chunks"]
        with pytest.raises(KeyError):
            create_selector({"strategy": "relative_threshold", "relative_threshold": incomplete})
