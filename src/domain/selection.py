"""Relative threshold selection strategy (Selector port).

Three-phase approach:
1. RANK: Sort by score (never by density/length)
2. FILTER: Apply relative thresholds (log_margin, drop_ratio, max_chunks)
3. PACK: Sequential by score until budget exhausted

Pure domain module with no external dependencies.
"""

from dataclasses import dataclass
from typing import Any

from src.domain.models import Chunk, ScoredChunk


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for RelativeThresholdSelector.

    Attributes:
        min_abs_score: Absolute noise floor
        log_margin: Relative floor (keep > s_max * 10^-log_margin)
        drop_ratio: Drop-off threshold between consecutive candidates
        max_chunks: Hard cap on retained chunks
        min_chunks: Minimum chunks (never empty context if candidates exist)
    """

    min_abs_score: float = 1e-3
    log_margin: float = 1.5
    drop_ratio: float = 0.20
    max_chunks: int = 5
    min_chunks: int = 1


class RelativeThresholdSelector:
    """Chunk selection by relative threshold to best score.

    Ranking and filtering depend only on score. Packing is sequential by score
    (no density sorting) to avoid sacrificing high-relevance chunks.
    """

    def __init__(self, config: SelectionConfig) -> None:
        self._config = config

    def select(self, query: str, chunks: list[ScoredChunk], context_budget: int) -> list[Chunk]:
        """Selectionne les chunks pertinents sous `context_budget`.

        Args:
            query: La question de l'utilisateur (non utilisee par cette
                strategie : la pertinence est deja portee par `score`).
            chunks: Chunks candidats, avec leur score de pertinence.
            context_budget: Budget de tokens disponible, jamais depasse.

        Returns:
            Les chunks selectionnes, dans l'ordre de pertinence (meilleur
            score en premier). Jamais vide si `chunks` est non vide.
        """
        cfg = self._config
        if not chunks:
            return []

        # --- PHASE 1 : ORDONNER (score pur, jamais la densite/longueur) ---
        ranked = sorted(chunks, key=lambda scored: scored.score, reverse=True)
        above_floor = [scored for scored in ranked if scored.score >= cfg.min_abs_score]
        ranked = above_floor or ranked[: cfg.min_chunks]

        # --- PHASE 2 : SELECTIONNER (relatif au meilleur score) ---
        s_max = ranked[0].score
        floor = s_max * (10 ** -cfg.log_margin)
        kept = [scored for scored in ranked if scored.score >= floor]

        for i in range(1, len(kept)):
            if kept[i].score < kept[i - 1].score * cfg.drop_ratio:
                kept = kept[:i]
                break
        kept = kept[: cfg.max_chunks]

        # --- PHASE 3 : PACKER (sequentiel par score) ---
        selected: list[ScoredChunk] = []
        total_tokens = 0
        for scored in kept:  # kept est deja trie par score decroissant
            if total_tokens + scored.chunk.n_tokens <= context_budget:
                selected.append(scored)
                total_tokens += scored.chunk.n_tokens

        fallback = ranked[: cfg.min_chunks]
        return [scored.chunk for scored in (selected or fallback)]

    def explain(self, chunks: list[ScoredChunk]) -> list[dict[str, Any]]:
        """Detaille, pour chaque candidat, pourquoi il est retenu ou non.

        Rejoue la meme logique que `select` (memes 3 phases), sans budget
        applique (`context_budget` infini) sauf pour le motif `"budget"` lui
        meme, qui necessite un budget explicite pour etre observe - cf. note
        ci-dessous. Utilise pour le rapport d'ablation (Etape 5).

        Args:
            chunks: Chunks candidats, avec leur score de pertinence.

        Returns:
            Une entree par candidat (ordre de score decroissant), avec les
            cles `score`, `n_tokens`, `rank` (1 = meilleur score), `source_document`
            (`chunk.source`), `kept` (bool) et `reason` parmi `"below_floor"`,
            `"drop_off"`, `"max_chunks"`, `"budget"`, `"kept"`. Le motif
            `"budget"` n'est jamais produit ici (aucun budget applique) : cette
            methode explique la SELECTION (phases 1-2), pas le PACKING
            (phase 3), qui depend d'un `context_budget` que `explain()` ne
            recoit pas.
        """
        cfg = self._config
        ranked = sorted(chunks, key=lambda scored: scored.score, reverse=True)

        entries: list[dict[str, Any]] = [
            {
                "score": scored.score,
                "n_tokens": scored.chunk.n_tokens,
                "rank": rank,
                "source_document": scored.chunk.source,
                "kept": False,
                "reason": "below_floor",
            }
            for rank, scored in enumerate(ranked, start=1)
        ]

        above_floor = [e for e in entries if e["score"] >= cfg.min_abs_score]
        candidates = above_floor or entries[: cfg.min_chunks]
        if not above_floor:
            for e in candidates:
                e["reason"] = "kept"

        if not candidates:
            return entries

        s_max = candidates[0]["score"]
        floor = s_max * (10 ** -cfg.log_margin)
        for e in candidates:
            if e["score"] < floor:
                e["reason"] = "below_floor"

        survivors = [e for e in candidates if e["score"] >= floor]
        for i in range(1, len(survivors)):
            if survivors[i]["score"] < survivors[i - 1]["score"] * cfg.drop_ratio:
                for e in survivors[i:]:
                    e["reason"] = "drop_off"
                survivors = survivors[:i]
                break

        for e in survivors[: cfg.max_chunks]:
            e["kept"] = True
            e["reason"] = "kept"
        for e in survivors[cfg.max_chunks :]:
            e["reason"] = "max_chunks"

        return entries
