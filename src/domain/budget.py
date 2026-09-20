"""Selection des chunks sous budget de tokens (port `Selector`).

Coeur du pipeline : decide quels chunks retrouves entrent effectivement dans
le contexte envoye au LLM, sous la contrainte stricte `context_budget`
(deja calculee par l'appelant = total - prompt - question - reserve_answer).

Module du domaine pur : aucune dependance externe (uniquement `math` de la
bibliotheque standard pour la similarite cosinus de MMR).

Convention pour `knapsack_mmr` : comme le port `Selector` ne transporte pas
les vecteurs d'embedding (seulement `ScoredChunk` = chunk + score), le
vecteur de chaque chunk est attendu dans `chunk.metadata["embedding"]`
(`list[float]`). C'est au code appelant (pipeline/adapters) de le renseigner
avant d'appeler `select` avec cette strategie.
"""

import math
from typing import Any

from src.domain.models import Chunk, ScoredChunk
from src.domain.ports import Selector

_DUPLICATE_SIMILARITY_THRESHOLD = 0.95

# Plancher de pertinence pour `KnapsackMMRSelector` (score de reranking brut,
# deja normalise en (0, 1) par le reranker). Un sigmoide(logit) ne redescend
# jamais exactement a 0.0 : pour un chunk hors-sujet, la valeur flottante
# reelle est de l'ordre de 1e-5 (bruit residuel), jamais 0.0 exactement - le
# test `score <= 0` ne l'excluait donc jamais. Constate en direct sur un
# corpus reel : chunks hors-sujet ~1.1e-5 a 1.9e-5, chunks pertinents mais
# faiblement scores (correspondance lexicale faible) ~0.012 a 0.013 - un
# ecart de ~3 ordres de grandeur. Ce seuil est place au milieu (sur echelle
# log) de cet ecart. Sans ce plancher, le tri par densite (score/n_tokens)
# favorise artificiellement le chunk le plus COURT parmi le bruit (numerateur
# quasi-nul, denominateur minimal = densite maximale), sans rapport avec sa
# pertinence reelle (bug reel constate : un chunk hors-sujet mais tres court
# evincait un chunk pertinent plus long qui ne rentrait plus dans le budget).
_MIN_RELEVANCE_SCORE = 1e-3


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


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similarite cosinus entre deux vecteurs (0.0 si l'un est nul)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embedding_of(chunk: Chunk) -> list[float]:
    """Recupere le vecteur d'embedding attache au chunk pour la strategie MMR.

    Raises:
        ValueError: Si `chunk.metadata["embedding"]` est absent.
    """
    embedding = chunk.metadata.get("embedding")
    if embedding is None:
        raise ValueError(
            f"le chunk {chunk.id!r} n'a pas d'embedding dans "
            "metadata['embedding'], requis par la strategie 'knapsack_mmr'"
        )
    return embedding


class TopKSelector:
    """Selectionne les chunks par score decroissant jusqu'a saturation du budget.

    Strategie naive : s'arrete au premier chunk qui ne rentre plus, sans
    chercher a combler le reste du budget avec des chunks plus petits plus
    loin dans le classement (contrairement a `KnapsackSelector`).
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


class KnapsackSelector:
    """Selectionne les chunks par densite (score / n_tokens) decroissante, gloutonne.

    Contrairement a `TopKSelector`, ne s'arrete pas au premier chunk qui ne
    rentre pas : continue d'examiner les suivants pour maximiser la valeur
    totale (somme des scores) sous la contrainte de budget.
    """

    def select(self, query: str, chunks: list[ScoredChunk], context_budget: int) -> list[Chunk]:
        """Selectionne les chunks maximisant la densite de pertinence sous budget.

        Args:
            query: La question de l'utilisateur (non utilisee par cette strategie).
            chunks: Chunks candidats, avec leur score de pertinence.
            context_budget: Budget de tokens disponible, jamais depasse.

        Returns:
            Les chunks selectionnes, reordonnes anti-lost-in-the-middle.
        """
        ordered = sorted(
            chunks, key=lambda scored: scored.score / scored.chunk.n_tokens, reverse=True
        )

        selected: list[Chunk] = []
        total_tokens = 0
        for scored in ordered:
            if total_tokens + scored.chunk.n_tokens > context_budget:
                continue
            selected.append(scored.chunk)
            total_tokens += scored.chunk.n_tokens

        return _anti_lost_in_middle(selected)


class KnapsackMMRSelector:
    """Selection par densite (comme `KnapsackSelector`), en penalisant la redondance.

    Avant d'ajouter un chunk, compare son embedding aux chunks deja
    selectionnes (Maximal Marginal Relevance) : rejette les quasi-duplicats
    (similarite cosinus > seuil) et les chunks sans pertinence propre
    (score brut < `_MIN_RELEVANCE_SCORE`).

    Note (correction d'un bug reel) : ne rejette PLUS un chunk a score positif
    sur la base d'un "score effectif" (mmr_lambda * score - (1 - mmr_lambda) *
    sim_max) devenu negatif. Avec un reranker cross-encoder, les scores sont
    souvent tres compresses pres de 0 (ex : 0.01) pour un chunk pertinent mais
    lexicalement eloigne de la requete (typique d'une question
    multi-documents). Dans ce regime, la moindre similarite cosinus avec un
    chunk deja selectionne (frequente entre documents d'un meme corpus, meme
    sans etre un vrai quasi-duplicat) suffisait a faire passer ce score
    effectif sous 0 et a rejeter un chunk pourtant pertinent, independamment
    du budget restant (bug reel constate : question multi-documents perdant
    une source pertinente malgre un budget largement disponible). Le filtrage
    anti-redondance reste assure par le rejet des quasi-duplicats
    (`sim_max > _DUPLICATE_SIMILARITY_THRESHOLD`) ; la pertinence minimale est
    assuree par le rejet des chunks a score brut < `_MIN_RELEVANCE_SCORE`
    (voir sa docstring : un sigmoide ne redescend jamais exactement a 0.0,
    un simple `score <= 0` ne filtrait donc jamais le bruit residuel).
    """

    def __init__(self, mmr_lambda: float) -> None:
        """Initialise le selecteur.

        Args:
            mmr_lambda: Conserve pour compatibilite avec la config
                (`selection.mmr_lambda`) mais n'est plus utilise dans la
                decision de selection (voir docstring de la classe) : l'ancien
                score effectif pondere par ce parametre causait un rejet
                errone de chunks pertinents a score faible.
        """
        self._mmr_lambda = mmr_lambda

    def select(self, query: str, chunks: list[ScoredChunk], context_budget: int) -> list[Chunk]:
        """Selectionne par densite sous budget, en filtrant la redondance (MMR).

        Args:
            query: La question de l'utilisateur (non utilisee : la redondance
                se mesure entre chunks, pas vis-a-vis de la requete).
            chunks: Chunks candidats, avec leur score de pertinence. Chaque
                chunk doit porter son embedding dans `metadata["embedding"]`.
            context_budget: Budget de tokens disponible, jamais depasse.

        Returns:
            Les chunks selectionnes, reordonnes anti-lost-in-the-middle.

        Raises:
            ValueError: Si un chunk n'a pas d'embedding dans `metadata["embedding"]`.
        """
        ordered = sorted(
            chunks, key=lambda scored: scored.score / scored.chunk.n_tokens, reverse=True
        )

        selected: list[Chunk] = []
        selected_embeddings: list[list[float]] = []
        total_tokens = 0

        for scored in ordered:
            if scored.score < _MIN_RELEVANCE_SCORE:
                continue

            embedding = _embedding_of(scored.chunk)
            sim_max = max(
                (_cosine_similarity(embedding, other) for other in selected_embeddings),
                default=0.0,
            )

            if sim_max > _DUPLICATE_SIMILARITY_THRESHOLD:
                continue

            if total_tokens + scored.chunk.n_tokens > context_budget:
                continue

            selected.append(scored.chunk)
            selected_embeddings.append(embedding)
            total_tokens += scored.chunk.n_tokens

        return _anti_lost_in_middle(selected)


def create_selector(config: dict[str, Any]) -> Selector:
    """Instancie le Selector configure (topk, knapsack, ou knapsack_mmr).

    Args:
        config: Section de configuration `selection`. Doit contenir `strategy`
            parmi {"topk", "knapsack", "knapsack_mmr"}. Si `strategy` vaut
            "knapsack_mmr", doit aussi contenir `mmr_lambda`.

    Returns:
        Une instance de `Selector` prete a l'emploi.

    Raises:
        KeyError: Si `strategy` est absent, ou si `mmr_lambda` est absent
            alors que `strategy` vaut "knapsack_mmr".
        ValueError: Si `strategy` ne correspond a aucune strategie connue.
    """
    try:
        strategy = config["strategy"]
    except KeyError as exc:
        raise KeyError("config de selection invalide : cle 'strategy' manquante") from exc

    if strategy == "topk":
        return TopKSelector()
    if strategy == "knapsack":
        return KnapsackSelector()
    if strategy == "knapsack_mmr":
        try:
            mmr_lambda = config["mmr_lambda"]
        except KeyError as exc:
            raise KeyError(
                "config de selection invalide : 'mmr_lambda' est requis pour 'knapsack_mmr'"
            ) from exc
        return KnapsackMMRSelector(mmr_lambda=mmr_lambda)

    raise ValueError(
        f"strategie de selection inconnue: {strategy!r} "
        "(attendu parmi 'topk', 'knapsack', 'knapsack_mmr')"
    )
