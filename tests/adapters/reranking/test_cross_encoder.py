"""Tests de CrossEncoderReranker : le reranking change l'ordre du retrieval brut.

Vrai modele cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2), pas de mock.
"""

from sentence_transformers import CrossEncoder

from src.adapters.reranking.cross_encoder import CrossEncoderReranker
from src.domain.models import Chunk, ScoredChunk

_QUERY = "What propellant powers the station's engines?"

# ordre "brut" deliberement mauvais : le chunk reellement pertinent (c1) est
# place en dernier, avec le score de retrieval le plus bas.
_RAW_ORDER = [
    ScoredChunk(
        chunk=Chunk(
            id="c2",
            text="The cafeteria menu today offers pasta and a green salad.",
            source="c2.md",
            n_tokens=10,
        ),
        score=0.9,
    ),
    ScoredChunk(
        chunk=Chunk(
            id="c3",
            text="Safety procedures list all emergency exits located on deck two.",
            source="c3.md",
            n_tokens=10,
        ),
        score=0.8,
    ),
    ScoredChunk(
        chunk=Chunk(
            id="c1",
            text="The propulsion system uses xenon as fuel for the ion thrusters.",
            source="c1.md",
            n_tokens=10,
        ),
        score=0.1,
    ),
]


class TestCrossEncoderReranker:
    def test_rerank_changes_order_vs_raw_retrieval(
        self, cross_encoder_model: CrossEncoder
    ) -> None:
        reranker = CrossEncoderReranker(model=cross_encoder_model)

        reranked = reranker.rerank(_QUERY, _RAW_ORDER, top_n=3)

        raw_order_ids = [sc.chunk.id for sc in _RAW_ORDER]
        reranked_ids = [sc.chunk.id for sc in reranked]

        assert reranked_ids != raw_order_ids
        assert reranked_ids[0] == "c1"

    def test_rerank_respects_top_n(self, cross_encoder_model: CrossEncoder) -> None:
        reranker = CrossEncoderReranker(model=cross_encoder_model)
        reranked = reranker.rerank(_QUERY, _RAW_ORDER, top_n=1)
        assert len(reranked) == 1
        assert reranked[0].chunk.id == "c1"

    def test_rerank_scores_are_sorted_descending(
        self, cross_encoder_model: CrossEncoder
    ) -> None:
        reranker = CrossEncoderReranker(model=cross_encoder_model)
        reranked = reranker.rerank(_QUERY, _RAW_ORDER, top_n=3)
        scores = [sc.score for sc in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_empty_chunks_returns_empty(self, cross_encoder_model: CrossEncoder) -> None:
        reranker = CrossEncoderReranker(model=cross_encoder_model)
        assert reranker.rerank(_QUERY, [], top_n=5) == []

    def test_scores_are_normalized_between_zero_and_one(
        self, cross_encoder_model: CrossEncoder
    ) -> None:
        """Le score brut du cross-encoder n'est pas borne (logit) ; une fois
        normalise, KnapsackMMRSelector peut evaluer un score effectif sans
        rejeter a tort un chunk pertinent (bug reel constate en production)."""
        reranker = CrossEncoderReranker(model=cross_encoder_model)
        # requete hors-sujet : le cross-encoder doit donner un score brut tres
        # negatif pour ces chunks, ce qui aurait pu produire un score final < 0
        # avant la normalisation.
        off_topic_query = "What is the capital of France?"
        reranked = reranker.rerank(off_topic_query, _RAW_ORDER, top_n=3)
        for sc in reranked:
            assert 0.0 <= sc.score <= 1.0
