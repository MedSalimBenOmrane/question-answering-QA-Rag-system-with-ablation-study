"""Strategie de selection `relative_threshold` (port `Selector`).

Remplace les anciennes strategies par densite (`knapsack`/`knapsack_mmr`,
retirees de `src/domain/budget.py`). Principe directeur : trois decisions
distinctes, jamais fusionnees dans un seul critere.

    PHASE 1 - ORDONNER     : par `score` pur (jamais la densite, jamais la
                              longueur) - la densite corrompt l'ordre de
                              PERTINENCE (un chunk long serait penalise
                              proportionnellement a sa longueur).
    PHASE 2 - SELECTIONNER : combien garder, par des criteres RELATIFS au
                              meilleur score (`log_margin`, `drop_ratio`,
                              `max_chunks`) - rend la decision independante
                              de l'echelle de score d'un reranker donne,
                              contrairement a un seuil absolu fixe.
    PHASE 3 - PACKER       : faire tenir dans le budget de tokens. La
                              densite (score / n_tokens) n'intervient QU'ICI,
                              pour maximiser la pertinence totale sous budget.

Module du domaine pur : aucune dependance externe.
"""

from dataclasses import dataclass
from typing import Any

from src.domain.models import Chunk, ScoredChunk


@dataclass(frozen=True)
class SelectionConfig:
    """Parametres numeriques de `RelativeThresholdSelector` (config-driven).

    Attributes:
        min_abs_score: Plancher de bruit absolu (calibrage herite de l'ancien
            `_MIN_RELEVANCE_SCORE` : un score de reranking en dessous est
            considere comme du bruit residuel, quel que soit le contexte).
        log_margin: Marge logarithmique appliquee au meilleur score pour
            obtenir le plancher RELATIF de la phase 2 (`floor = s_max *
            10 ** -log_margin`) : ex. 1.5 conserve les scores superieurs a
            environ 1/32e du meilleur score.
        drop_ratio: Seuil de decrochage entre deux candidats consecutifs
            (tries par score) : si `score[i] < score[i-1] * drop_ratio`, tous
            les candidats a partir du rang i sont ecartes.
        max_chunks: Plafond dur sur le nombre de chunks retenus a l'issue de
            la phase 2 (avant le packing sous budget de la phase 3).
        min_chunks: Nombre minimal de chunks a renvoyer : le selecteur ne
            renvoie jamais un contexte vide tant qu'au moins un candidat a
            ete fourni en entree.
    """

    min_abs_score: float = 1e-3
    log_margin: float = 1.5
    drop_ratio: float = 0.20
    max_chunks: int = 5
    min_chunks: int = 1


class RelativeThresholdSelector:
    """Selectionne les chunks par seuil RELATIF au meilleur score (cf. docstring du module).

    Contrairement aux strategies par densite retirees, le classement (phase 1)
    et le critere d'inclusion (phase 2) ne dependent jamais de `n_tokens` : la
    densite ne sert qu'a maximiser l'usage du budget restant (phase 3), jamais
    a decider QUELS chunks sont pertinents.
    """

    def __init__(self, config: SelectionConfig) -> None:
        """Initialise le selecteur.

        Args:
            config: Parametres numeriques (cf. `SelectionConfig`).
        """
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

        # --- PHASE 3 : PACKER (densite UNIQUEMENT ici) ---
        selected: list[ScoredChunk] = []
        total_tokens = 0
        for scored in sorted(
            kept, key=lambda scored: scored.score / max(scored.chunk.n_tokens, 1), reverse=True
        ):
            if total_tokens + scored.chunk.n_tokens <= context_budget:
                selected.append(scored)
                total_tokens += scored.chunk.n_tokens

        # Ordre du prompt = ordre de pertinence (score), pas l'ordre de packing.
        selected.sort(key=lambda scored: scored.score, reverse=True)
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
