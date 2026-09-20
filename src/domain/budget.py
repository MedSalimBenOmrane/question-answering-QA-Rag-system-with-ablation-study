"""Selection des chunks sous budget de tokens (port `Selector`).

Coeur du pipeline : decide quels chunks retrouves entrent effectivement dans
le contexte envoye au LLM, sous la contrainte stricte `context_budget`
(deja calculee par l'appelant = total - prompt - question - reserve_answer).

Module du domaine pur : aucune dependance externe.

Note (chantier "relative_threshold") : les strategies par densite avec
anti-redondance ont ete retirees de ce module. `TopKSelector` reste ici ;
la nouvelle strategie `relative_threshold` vit dans `src/domain/selection.py`
et est cablee par `create_selector` ci-dessous.
"""

from typing import Any

from src.domain.models import Chunk, ScoredChunk
from src.domain.ports import Selector
from src.domain.selection import RelativeThresholdSelector, SelectionConfig


def _anti_lost_in_middle(selected: list[Chunk]) -> list[Chunk]:
    """Reordonne pour placer le meilleur en tete et le 2e meilleur en queue.

    Attenue l'effet "lost-in-the-middle" des LLM (moins attentifs au milieu
    d'un long contexte) : le reste du classement (3e, 4e, ...) est place au
    milieu, dans son ordre relatif d'origine.

    Args:
        selected: Chunks dans l'ordre de selection (le meilleur en premier).

    Returns:
        Les memes chunks reordonnes : [meilleur, 3e, 4e, ..., dernier, 2e].
    """
    if len(selected) <= 2:
        return selected
    return [selected[0], *selected[2:], selected[1]]


class TopKSelector:
    """Selectionne les chunks par score decroissant jusqu'a saturation du budget.

    Strategie naive : s'arrete au premier chunk qui ne rentre plus, sans
    chercher a combler le reste du budget avec des chunks plus petits plus
    loin dans le classement.
    """

    def select(self, query: str, chunks: list[ScoredChunk], context_budget: int) -> list[Chunk]:
        """Selectionne les chunks les plus pertinents sous `context_budget`.

        Args:
            query: La question de l'utilisateur (non utilisee par cette strategie).
            chunks: Chunks candidats, avec leur score de pertinence.
            context_budget: Budget de tokens disponible, jamais depasse.

        Returns:
            Les chunks selectionnes, reordonnes anti-lost-in-the-middle.
        """
        ordered = sorted(chunks, key=lambda scored: scored.score, reverse=True)

        selected: list[Chunk] = []
        total_tokens = 0
        for scored in ordered:
            if total_tokens + scored.chunk.n_tokens > context_budget:
                break
            selected.append(scored.chunk)
            total_tokens += scored.chunk.n_tokens

        return _anti_lost_in_middle(selected)


_RELATIVE_THRESHOLD_KEYS = ("min_abs_score", "log_margin", "drop_ratio", "max_chunks", "min_chunks")


def create_selector(config: dict[str, Any]) -> Selector:
    """Instancie le Selector configure (topk ou relative_threshold).

    Args:
        config: Section de configuration `selection`. Doit contenir `strategy`
            parmi {"topk", "relative_threshold"}. Si `strategy` vaut
            "relative_threshold", doit aussi contenir une sous-section
            `relative_threshold` avec les 5 cles de `SelectionConfig`
            (`min_abs_score`, `log_margin`, `drop_ratio`, `max_chunks`,
            `min_chunks`).

    Returns:
        Une instance de `Selector` prete a l'emploi.

    Raises:
        KeyError: Si `strategy` est absent, ou si la sous-section/une cle
            `relative_threshold` est absente alors que `strategy` vaut
            "relative_threshold".
        ValueError: Si `strategy` ne correspond a aucune strategie connue.
    """
    try:
        strategy = config["strategy"]
    except KeyError as exc:
        raise KeyError("config de selection invalide : cle 'strategy' manquante") from exc

    if strategy == "topk":
        return TopKSelector()

    if strategy == "relative_threshold":
        try:
            rt_config = config["relative_threshold"]
        except KeyError as exc:
            raise KeyError(
                "config de selection invalide : section 'relative_threshold' "
                "requise pour la strategie 'relative_threshold'"
            ) from exc
        try:
            selection_config = SelectionConfig(**{key: rt_config[key] for key in _RELATIVE_THRESHOLD_KEYS})
        except KeyError as exc:
            raise KeyError(
                f"config de selection invalide : cle {exc.args[0]!r} manquante dans "
                "'selection.relative_threshold' (requis : "
                f"{', '.join(_RELATIVE_THRESHOLD_KEYS)})"
            ) from exc
        return RelativeThresholdSelector(selection_config)

    raise ValueError(
        f"strategie de selection inconnue: {strategy!r} "
        "(attendu parmi 'topk', 'relative_threshold')"
    )
