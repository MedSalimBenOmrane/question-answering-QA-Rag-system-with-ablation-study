"""Tests de src/domain/budget.py (port Selector) : topk, knapsack, knapsack_mmr.

Coeur du projet : ces tests verifient rigoureusement le respect du budget de
tokens (jamais depasse), la difference de comportement topk/knapsack, et le
filtrage des quasi-duplicats par MMR.
"""

import math

import pytest

from src.adapters.embedding.bge_m3 import BgeM3Embedder
from src.domain.budget import KnapsackMMRSelector, KnapsackSelector, TopKSelector, create_selector
from src.domain.models import Chunk, ScoredChunk


def _chunk(chunk_id: str, n_tokens: int, embedding: list[float] | None = None) -> Chunk:
    metadata = {"embedding": embedding} if embedding is not None else {}
    return Chunk(id=chunk_id, text=f"text of {chunk_id}", source="test.md", n_tokens=n_tokens, metadata=metadata)


def _total_tokens(chunks: list[Chunk]) -> int:
    return sum(c.n_tokens for c in chunks)


class TestBudgetNeverExceeded:
    """Invariant central : la somme des n_tokens selectionnes ne depasse jamais le budget."""

    CHUNKS = [
        ScoredChunk(chunk=_chunk("a", 350, [1.0, 0.0, 0.0]), score=0.9),
        ScoredChunk(chunk=_chunk("b", 150, [0.0, 1.0, 0.0]), score=0.8),
        ScoredChunk(chunk=_chunk("c", 150, [0.0, 0.0, 1.0]), score=0.75),
        ScoredChunk(chunk=_chunk("d", 140, [1.0, 1.0, 0.0]), score=0.7),
    ]
    BUDGET = 600

    def test_topk_never_exceeds_budget(self) -> None:
        result = TopKSelector().select("q", self.CHUNKS, self.BUDGET)
        assert _total_tokens(result) <= self.BUDGET

    def test_knapsack_never_exceeds_budget(self) -> None:
        result = KnapsackSelector().select("q", self.CHUNKS, self.BUDGET)
        assert _total_tokens(result) <= self.BUDGET

    def test_knapsack_mmr_never_exceeds_budget(self) -> None:
        result = KnapsackMMRSelector(mmr_lambda=0.7).select("q", self.CHUNKS, self.BUDGET)
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

    def test_knapsack_includes_chunk_that_exactly_fills_remaining_budget(self) -> None:
        chunks = [
            ScoredChunk(chunk=_chunk("a", 300), score=1.0),
            ScoredChunk(chunk=_chunk("b", 300), score=0.9),
        ]
        result = KnapsackSelector().select("q", chunks, context_budget=600)

        assert {c.id for c in result} == {"a", "b"}
        assert _total_tokens(result) == 600


class TestBudgetZero:
    def test_topk_budget_zero_returns_empty(self) -> None:
        chunks = [ScoredChunk(chunk=_chunk("a", 10), score=0.9)]
        assert TopKSelector().select("q", chunks, context_budget=0) == []

    def test_knapsack_budget_zero_returns_empty(self) -> None:
        chunks = [ScoredChunk(chunk=_chunk("a", 10), score=0.9)]
        assert KnapsackSelector().select("q", chunks, context_budget=0) == []

    def test_knapsack_mmr_budget_zero_returns_empty(self) -> None:
        chunks = [ScoredChunk(chunk=_chunk("a", 10, [1.0, 0.0]), score=0.9)]
        assert KnapsackMMRSelector(mmr_lambda=0.7).select("q", chunks, context_budget=0) == []


class TestSingleChunkTooBig:
    def test_topk_single_oversized_chunk_returns_empty(self) -> None:
        chunks = [ScoredChunk(chunk=_chunk("a", 1000), score=0.9)]
        assert TopKSelector().select("q", chunks, context_budget=600) == []

    def test_knapsack_single_oversized_chunk_returns_empty(self) -> None:
        chunks = [ScoredChunk(chunk=_chunk("a", 1000), score=0.9)]
        assert KnapsackSelector().select("q", chunks, context_budget=600) == []

    def test_knapsack_mmr_single_oversized_chunk_returns_empty(self) -> None:
        chunks = [ScoredChunk(chunk=_chunk("a", 1000, [1.0, 0.0]), score=0.9)]
        assert KnapsackMMRSelector(mmr_lambda=0.7).select("q", chunks, context_budget=600) == []


class TestKnapsackBeatsTopK:
    """Reproduit l'exemple impose : A=0.9/350, B=0.8/150, C=0.75/150, D=0.7/140, budget=600."""

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

    def test_knapsack_picks_b_c_d(self) -> None:
        result = KnapsackSelector().select("q", self.CHUNKS, self.BUDGET)
        assert {c.id for c in result} == {"B", "C", "D"}
        assert _total_tokens(result) == 440

    def test_knapsack_achieves_higher_total_relevance_than_topk(self) -> None:
        topk_score = sum(sc.score for sc in self.CHUNKS if sc.chunk.id in {"A", "B"})
        knapsack_score = sum(sc.score for sc in self.CHUNKS if sc.chunk.id in {"B", "C", "D"})
        assert knapsack_score > topk_score


class TestKnapsackMMRRemovesDuplicate:
    """MMR ecarte un chunk quasi-duplicat (similarite cosinus > 0.95) deja selectionne."""

    def test_near_duplicate_is_excluded(self) -> None:
        v1 = [1.0, 0.0]
        # cosinus(v1, v2) = 0.99 > 0.95 : quasi-duplicat de v1
        v2 = [0.99, math.sqrt(1 - 0.99**2)]
        v3 = [0.0, 1.0]  # orthogonal a v1 : pas un duplicat

        chunks = [
            ScoredChunk(chunk=_chunk("dup1", 100, v1), score=0.9),
            ScoredChunk(chunk=_chunk("dup2", 100, v2), score=0.85),
            ScoredChunk(chunk=_chunk("other", 100, v3), score=0.5),
        ]

        result = KnapsackMMRSelector(mmr_lambda=0.7).select("q", chunks, context_budget=300)
        ids = {c.id for c in result}

        assert "dup1" in ids
        assert "dup2" not in ids
        assert "other" in ids

    def test_missing_embedding_raises(self) -> None:
        chunks = [ScoredChunk(chunk=_chunk("no_embedding", 100), score=0.9)]
        with pytest.raises(ValueError):
            KnapsackMMRSelector(mmr_lambda=0.7).select("q", chunks, context_budget=300)

    def test_real_embeddings_duplicate_text_is_deduplicated(
        self, bge_m3_embedder: BgeM3Embedder
    ) -> None:
        text_a = "The propulsion system uses xenon as fuel for the ion thrusters."
        text_b = text_a  # duplicat exact -> similarite ~1.0 avec de vrais embeddings
        text_c = "The cafeteria menu today offers pasta and a green salad."

        [vec_a, vec_b, vec_c] = bge_m3_embedder.embed([text_a, text_b, text_c])

        chunks = [
            ScoredChunk(chunk=_chunk("a", 10, vec_a), score=0.9),
            ScoredChunk(chunk=_chunk("b", 10, vec_b), score=0.88),
            ScoredChunk(chunk=_chunk("c", 10, vec_c), score=0.4),
        ]

        result = KnapsackMMRSelector(mmr_lambda=0.7).select("q", chunks, context_budget=30)
        ids = {c.id for c in result}

        assert "a" in ids
        assert "b" not in ids
        assert "c" in ids


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


class TestCreateSelector:
    def test_creates_topk(self) -> None:
        assert isinstance(create_selector({"strategy": "topk"}), TopKSelector)

    def test_creates_knapsack(self) -> None:
        assert isinstance(create_selector({"strategy": "knapsack"}), KnapsackSelector)

    def test_creates_knapsack_mmr(self) -> None:
        selector = create_selector({"strategy": "knapsack_mmr", "mmr_lambda": 0.7})
        assert isinstance(selector, KnapsackMMRSelector)

    def test_missing_strategy_raises(self) -> None:
        with pytest.raises(KeyError):
            create_selector({})

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError):
            create_selector({"strategy": "does-not-exist"})

    def test_missing_mmr_lambda_raises(self) -> None:
        with pytest.raises(KeyError):
            create_selector({"strategy": "knapsack_mmr"})
