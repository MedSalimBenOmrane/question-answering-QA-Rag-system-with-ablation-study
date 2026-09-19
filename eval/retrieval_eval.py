"""Evalue le retrieval configure contre un gold set fait main (Recall@k,
Precision@k, MRR, nDCG@k), puis ecrit un rapport Markdown + CSV.

La pertinence est jugee au niveau SOURCE (`relevant_sources` du gold set),
pas au niveau chunk : un chunk retrouve est considere pertinent si
`chunk.source` figure dans `relevant_sources` de la question, quel que soit
le chunk precis.

Formules (binaire, coupees a k) :
- Precision@k = (# chunks pertinents dans le top-k) / k
- Recall@k = (# sources pertinentes distinctes retrouvees dans le top-k)
  / (# sources pertinentes attendues)
- RR (reciprocal rank) = 1 / rang du premier chunk pertinent dans le top-k
  (0 si aucun) ; MRR = moyenne des RR sur toutes les questions.
- nDCG@k = DCG@k / IDCG@k, avec DCG@k = somme_{i=1..k} rel_i / log2(i+1)
  (rel_i binaire, par chunk). IDCG@k suppose une source pertinente par rang
  ideal : somme_{i=1..min(k, n_relevant_sources)} 1 / log2(i+1).

Format attendu de `eval/gold_retrieval.yaml` (liste d'items) :
    - id: q1
      question: "..."
      type: multi_doc          # optionnel, informatif
      relevant_sources:
        - boot_sequence.md
        - long_procedures.md
      key_points:               # optionnel, non utilise ici (cf. generation_eval.py)
        - "..."

Usage:
    uv run python -m eval.retrieval_eval
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.domain.ports import Retriever

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_GOLD_PATH = _REPO_ROOT / "eval" / "gold_retrieval.yaml"
_DEFAULT_RESULTS_CSV_PATH = _REPO_ROOT / "eval" / "retrieval_eval_results.csv"


@dataclass
class GoldItem:
    """Un item du gold set retrieval.

    Attributes:
        id: Identifiant court de la question (utilise dans les rapports).
        question: La question posee.
        relevant_sources: Sources jugees pertinentes pour y repondre.
        type: Categorie informative de la question (ex: "multi_doc"),
            optionnelle, non utilisee dans le calcul des metriques.
    """

    id: str
    question: str
    relevant_sources: list[str]
    type: str | None = None


@dataclass
class QueryMetrics:
    """Metriques de retrieval calculees pour une question."""

    id: str
    question: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


def load_gold_set(path: Path = _DEFAULT_GOLD_PATH) -> list[GoldItem]:
    """Charge le gold set d'evaluation retrieval.

    Args:
        path: Chemin du fichier YAML (liste d'items, cf. docstring du module).

    Returns:
        La liste des items du gold set.

    Raises:
        ValueError: Si le fichier est vide, mal forme, ou si un item n'a
            aucune source pertinente.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not raw:
        raise ValueError(f"gold set vide ou introuvable : {path}")

    items: list[GoldItem] = []
    for entry in raw:
        try:
            relevant_sources = list(entry["relevant_sources"])
            if not relevant_sources:
                raise ValueError(f"item {entry.get('id')!r} : 'relevant_sources' est vide")
            items.append(
                GoldItem(
                    id=entry["id"],
                    question=entry["question"],
                    relevant_sources=relevant_sources,
                    type=entry.get("type"),
                )
            )
        except KeyError as exc:
            raise ValueError(
                f"item de gold set invalide (id/question/relevant_sources requis) : {entry}"
            ) from exc

    return items


def _precision_at_k(retrieved_sources: list[str], relevant_sources: set[str], k: int) -> float:
    """Precision@k : proportion de chunks pertinents dans le top-k."""
    if k <= 0:
        return 0.0
    top_k = retrieved_sources[:k]
    hits = sum(1 for source in top_k if source in relevant_sources)
    return hits / k


def _recall_at_k(retrieved_sources: list[str], relevant_sources: set[str], k: int) -> float:
    """Recall@k : proportion de sources pertinentes distinctes retrouvees dans le top-k."""
    if not relevant_sources:
        return 0.0
    top_k = retrieved_sources[:k]
    found = {source for source in top_k if source in relevant_sources}
    return len(found) / len(relevant_sources)


def _reciprocal_rank(retrieved_sources: list[str], relevant_sources: set[str], k: int) -> float:
    """1 / rang du premier chunk pertinent dans le top-k (0 si aucun)."""
    for rank, source in enumerate(retrieved_sources[:k], start=1):
        if source in relevant_sources:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(retrieved_sources: list[str], relevant_sources: set[str], k: int) -> float:
    """nDCG@k avec pertinence binaire par chunk (cf. docstring du module)."""
    top_k = retrieved_sources[:k]
    dcg = sum(
        (1.0 if source in relevant_sources else 0.0) / math.log2(rank + 1)
        for rank, source in enumerate(top_k, start=1)
    )
    ideal_hits = min(k, len(relevant_sources))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_query(retriever: Retriever, item: GoldItem, k: int) -> QueryMetrics:
    """Lance le retrieval pour une question du gold set et calcule ses metriques.

    Args:
        retriever: Le Retriever a evaluer (deja construit/configure).
        item: L'item du gold set (question + sources pertinentes attendues).
        k: Rang de coupure applique a toutes les metriques.

    Returns:
        Les metriques calculees pour cette question.
    """
    results = retriever.retrieve(item.question, k)
    retrieved_sources = [scored.chunk.source for scored in results]
    relevant_sources = set(item.relevant_sources)

    return QueryMetrics(
        id=item.id,
        question=item.question,
        precision_at_k=_precision_at_k(retrieved_sources, relevant_sources, k),
        recall_at_k=_recall_at_k(retrieved_sources, relevant_sources, k),
        reciprocal_rank=_reciprocal_rank(retrieved_sources, relevant_sources, k),
        ndcg_at_k=_ndcg_at_k(retrieved_sources, relevant_sources, k),
    )


def evaluate(retriever: Retriever, gold_set: list[GoldItem], k: int) -> list[QueryMetrics]:
    """Evalue le retrieval sur l'ensemble du gold set.

    Args:
        retriever: Le Retriever a evaluer.
        gold_set: Les items du gold set (cf. `load_gold_set`).
        k: Rang de coupure applique a toutes les metriques.

    Returns:
        Les metriques par question, dans l'ordre du gold set.
    """
    return [evaluate_query(retriever, item, k) for item in gold_set]


def summarize(results: list[QueryMetrics]) -> dict[str, float]:
    """Moyenne les metriques sur toutes les questions (MRR = moyenne des RR).

    Args:
        results: Les metriques par question (cf. `evaluate`).

    Returns:
        Un dict {precision_at_k, recall_at_k, mrr, ndcg_at_k} moyenne.
    """
    n = len(results)
    if n == 0:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}

    return {
        "precision_at_k": sum(r.precision_at_k for r in results) / n,
        "recall_at_k": sum(r.recall_at_k for r in results) / n,
        "mrr": sum(r.reciprocal_rank for r in results) / n,
        "ndcg_at_k": sum(r.ndcg_at_k for r in results) / n,
    }


def to_markdown_table(results: list[QueryMetrics], k: int, summary: dict[str, float]) -> str:
    """Formate les resultats (par question + moyenne) en tableau Markdown."""
    lines = [
        f"| Question ID | Precision@{k} | Recall@{k} | RR | nDCG@{k} |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.id} | {r.precision_at_k:.3f} | {r.recall_at_k:.3f} | "
            f"{r.reciprocal_rank:.3f} | {r.ndcg_at_k:.3f} |"
        )
    lines.append(
        f"| **Moyenne** | **{summary['precision_at_k']:.3f}** | "
        f"**{summary['recall_at_k']:.3f}** | **{summary['mrr']:.3f}** | "
        f"**{summary['ndcg_at_k']:.3f}** |"
    )
    return "\n".join(lines)


def write_csv(
    results: list[QueryMetrics], k: int, summary: dict[str, float], path: Path
) -> None:
    """Ecrit les resultats (par question + moyenne) en CSV.

    Args:
        results: Les metriques par question.
        k: Rang de coupure (pour l'entete des colonnes).
        summary: Les metriques moyennes (cf. `summarize`).
        path: Fichier CSV de sortie (cree/ecrase).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["question_id", f"precision_at_{k}", f"recall_at_{k}", "rr", f"ndcg_at_{k}"]
        )
        for r in results:
            writer.writerow(
                [r.id, r.precision_at_k, r.recall_at_k, r.reciprocal_rank, r.ndcg_at_k]
            )
        writer.writerow(
            [
                "MEAN",
                summary["precision_at_k"],
                summary["recall_at_k"],
                summary["mrr"],
                summary["ndcg_at_k"],
            ]
        )


def main() -> None:
    """Charge la config + le gold set reels, evalue le retrieval, ecrit les rapports."""
    from src.adapters.embedding.factory import create_embedder
    from src.adapters.retrieval.factory import create_retriever
    from src.adapters.vectorstore.chroma import create_vectorstore
    from src.cli import load_chunks, load_config

    config: dict[str, Any] = load_config()

    chunks_path = Path(config["vectorstore"]["chunks_path"])
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"index introuvable ({chunks_path}) : executer "
            "`uv run python -m src.cli index` avant l'evaluation"
        )
    chunks = load_chunks(chunks_path)

    embedder = create_embedder(config["embedding"])
    vectorstore = create_vectorstore(config["vectorstore"])
    retriever = create_retriever(config["retrieval"], embedder, vectorstore, chunks)

    gold_set = load_gold_set()
    k = config["pipeline"]["retrieve_k"]

    results = evaluate(retriever, gold_set, k)
    summary = summarize(results)

    print(to_markdown_table(results, k, summary))

    write_csv(results, k, summary, _DEFAULT_RESULTS_CSV_PATH)
    print(f"\nResultats CSV ecrits dans {_DEFAULT_RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
