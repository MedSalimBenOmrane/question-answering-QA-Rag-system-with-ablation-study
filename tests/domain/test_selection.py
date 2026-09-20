"""Tests de src/domain/selection.py (RelativeThresholdSelector, port Selector).

Aucun mock. Couvre les 6 scenarios explicitement requis par le chantier
"relative_threshold" : candidat unique, scores plats + plafond, decrochage
net, tous sous le plancher, non-regression densite/ordre, et packing sans
arret premature sous budget sature.
"""

from src.domain.models import Chunk, ScoredChunk
from src.domain.selection import RelativeThresholdSelector, SelectionConfig


def _chunk(chunk_id: str, n_tokens: int) -> Chunk:
    return Chunk(id=chunk_id, text=f"text of {chunk_id}", source=f"{chunk_id}.md", n_tokens=n_tokens)


class TestSingleStronglyRelevantCandidate:
    def test_returns_exactly_one_chunk(self) -> None:
        chunks = [ScoredChunk(chunk=_chunk("a", 50), score=0.9)]
        selector = RelativeThresholdSelector(SelectionConfig())

        result = selector.select("q", chunks, context_budget=1000)

        assert [c.id for c in result] == ["a"]


class TestFlatScoresCapAppliedWithoutDropOff:
    """Scores plats (0.5, 0.48, 0.47, 0.46) : aucun decrochage ne se declenche
    (chaque score reste bien au-dessus de score_precedent * drop_ratio), donc
    c'est le plafond max_chunks - pas un decrochage - qui limite le resultat."""

    CHUNKS = [
        ScoredChunk(chunk=_chunk("a", 10), score=0.50),
        ScoredChunk(chunk=_chunk("b", 10), score=0.48),
        ScoredChunk(chunk=_chunk("c", 10), score=0.47),
        ScoredChunk(chunk=_chunk("d", 10), score=0.46),
    ]

    def test_cap_limits_result_not_drop_off(self) -> None:
        cfg = SelectionConfig(max_chunks=3)
        selector = RelativeThresholdSelector(cfg)

        result = selector.select("q", self.CHUNKS, context_budget=1000)

        assert [c.id for c in result] == ["a", "b", "c"]

    def test_no_cap_keeps_all_four(self) -> None:
        cfg = SelectionConfig(max_chunks=5)
        selector = RelativeThresholdSelector(cfg)

        result = selector.select("q", self.CHUNKS, context_budget=1000)

        assert {c.id for c in result} == {"a", "b", "c", "d"}


class TestSharpDropOff:
    def test_sharp_drop_off_keeps_two_chunks(self) -> None:
        chunks = [
            ScoredChunk(chunk=_chunk("a", 10), score=0.9),
            ScoredChunk(chunk=_chunk("b", 10), score=0.85),
            ScoredChunk(chunk=_chunk("c", 10), score=0.02),
        ]
        selector = RelativeThresholdSelector(SelectionConfig())

        result = selector.select("q", chunks, context_budget=1000)

        assert [c.id for c in result] == ["a", "b"]


class TestAllBelowMinAbsScore:
    def test_never_returns_empty_returns_min_chunks(self) -> None:
        chunks = [
            ScoredChunk(chunk=_chunk("a", 10), score=1e-6),
            ScoredChunk(chunk=_chunk("b", 10), score=1e-7),
            ScoredChunk(chunk=_chunk("c", 10), score=1e-8),
        ]
        selector = RelativeThresholdSelector(SelectionConfig(min_chunks=1))

        result = selector.select("q", chunks, context_budget=1000)

        assert [c.id for c in result] == ["a"]

    def test_min_chunks_two_is_not_a_hard_floor_past_phase_2(self) -> None:
        """Nuance a signaler : `min_chunks` (cf. sa docstring/le commentaire de
        la spec : "ne jamais renvoyer un contexte vide") garantit seulement un
        resultat NON VIDE, pas "au moins N chunks" pour N > 1. Ici, le
        fallback de phase 1 repeche bien 2 candidats (a, b) malgre un score
        sous min_abs_score, mais le decrochage de phase 2 (drop_ratio) peut
        ensuite en retirer un (b : 1e-7 < 1e-6 * 0.20 = 2e-7) - le resultat
        final n'a alors qu'1 chunk, pas 2, sans jamais etre vide."""
        chunks = [
            ScoredChunk(chunk=_chunk("a", 10), score=1e-6),
            ScoredChunk(chunk=_chunk("b", 10), score=1e-7),
            ScoredChunk(chunk=_chunk("c", 10), score=1e-8),
        ]
        selector = RelativeThresholdSelector(SelectionConfig(min_chunks=2))

        result = selector.select("q", chunks, context_budget=1000)

        assert [c.id for c in result] == ["a"]
        assert len(result) >= 1  # jamais vide, seule garantie litterale de min_chunks


class TestLongChunkWithBestScoreIsFirst:
    """Non-regression contre le bug de densite : PHASE 1 ordonne par score pur,
    la densite (PHASE 3) ne sert qu'a packer, jamais a decider l'ordre final."""

    def test_long_best_scored_chunk_is_first_in_output(self) -> None:
        chunk_a = ScoredChunk(chunk=_chunk("long_best", 500), score=0.9)
        chunk_b = ScoredChunk(chunk=_chunk("short_worse", 50), score=0.5)

        selector = RelativeThresholdSelector(SelectionConfig())
        result = selector.select("q", [chunk_a, chunk_b], context_budget=600)

        assert [c.id for c in result] == ["long_best", "short_worse"]


class TestBudgetPackingNoEarlyBreak:
    """Budget sature : un chunk plus court plus loin dans le classement doit
    encore pouvoir entrer (pas de `break` premature comme TopKSelector)."""

    def test_shorter_chunk_further_down_still_included(self) -> None:
        chunk_a = ScoredChunk(chunk=_chunk("a", 400), score=0.9)
        chunk_b = ScoredChunk(chunk=_chunk("b", 400), score=0.8)
        chunk_c = ScoredChunk(chunk=_chunk("c", 50), score=0.7)

        selector = RelativeThresholdSelector(SelectionConfig())
        result = selector.select("q", [chunk_a, chunk_b, chunk_c], context_budget=450)

        # b (400 tokens) ne rentre pas apres a (400 tokens, budget=450) ; c
        # (50 tokens, rang 3, score le plus faible) rentre quand meme.
        assert [c.id for c in result] == ["a", "c"]


class TestSelectEdgeCases:
    def test_empty_input_returns_empty(self) -> None:
        selector = RelativeThresholdSelector(SelectionConfig())
        assert selector.select("q", [], context_budget=1000) == []

    def test_never_exceeds_budget(self) -> None:
        chunks = [
            ScoredChunk(chunk=_chunk("a", 350), score=0.9),
            ScoredChunk(chunk=_chunk("b", 150), score=0.8),
            ScoredChunk(chunk=_chunk("c", 150), score=0.75),
        ]
        selector = RelativeThresholdSelector(SelectionConfig())

        result = selector.select("q", chunks, context_budget=600)

        assert sum(c.n_tokens for c in result) <= 600


class TestExplain:
    def test_reasons_match_select_outcome(self) -> None:
        chunks = [
            ScoredChunk(chunk=_chunk("a", 10), score=0.9),
            ScoredChunk(chunk=_chunk("b", 10), score=0.85),
            ScoredChunk(chunk=_chunk("c", 10), score=0.02),
        ]
        selector = RelativeThresholdSelector(SelectionConfig())

        entries = selector.explain(chunks)
        by_id = {e["source_document"]: e for e in entries}

        assert by_id["a.md"]["kept"] is True
        assert by_id["a.md"]["reason"] == "kept"
        assert by_id["a.md"]["rank"] == 1
        assert by_id["b.md"]["kept"] is True
        assert by_id["c.md"]["kept"] is False
        assert by_id["c.md"]["reason"] == "below_floor"

    def test_max_chunks_reason(self) -> None:
        chunks = [
            ScoredChunk(chunk=_chunk("a", 10), score=0.50),
            ScoredChunk(chunk=_chunk("b", 10), score=0.48),
            ScoredChunk(chunk=_chunk("c", 10), score=0.47),
            ScoredChunk(chunk=_chunk("d", 10), score=0.46),
        ]
        selector = RelativeThresholdSelector(SelectionConfig(max_chunks=3))

        entries = selector.explain(chunks)
        by_id = {e["source_document"]: e for e in entries}

        assert by_id["d.md"]["kept"] is False
        assert by_id["d.md"]["reason"] == "max_chunks"

    def test_entries_cover_every_candidate(self) -> None:
        chunks = [
            ScoredChunk(chunk=_chunk("a", 10), score=0.9),
            ScoredChunk(chunk=_chunk("b", 10), score=1e-8),
        ]
        selector = RelativeThresholdSelector(SelectionConfig())

        entries = selector.explain(chunks)

        assert len(entries) == 2
        assert {e["n_tokens"] for e in entries} == {10}


class TestSelectionConfigDefaults:
    def test_defaults_match_spec(self) -> None:
        cfg = SelectionConfig()
        assert cfg.min_abs_score == 1e-3
        assert cfg.log_margin == 1.5
        assert cfg.drop_ratio == 0.20
        assert cfg.max_chunks == 5
        assert cfg.min_chunks == 1

    def test_is_frozen(self) -> None:
        cfg = SelectionConfig()
        try:
            cfg.max_chunks = 10  # type: ignore[misc]
        except Exception:
            pass
        else:
            raise AssertionError("SelectionConfig doit etre immuable (frozen=True)")
