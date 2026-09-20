"""Evalue le retrieval + la selection du pipeline RAG contre le gold set,
sur la SORTIE FINALE du pipeline (chantier "relative_threshold", etape 3).

Toutes les metriques sont calculees au niveau SOURCE (nom de fichier), jamais
au niveau chunk : le gold liste des noms de fichiers (`relevant_sources`), et
une eval au niveau chunk rendrait des strategies de chunking differentes
incomparables (cf. `to_sources`).

Quatre metriques, chacune sur l'etage du pipeline qui lui correspond :
- `candidate_recall` (garde-fou) : sur les candidats du RETRIEVER, AVANT
  reranking. S'il est < 1.0, une source manquante n'a jamais atteint le
  reranker - aucune amelioration du selector ne peut alors la recuperer.
- `ndcg_at_10` : sur l'ordre du RERANKER, AVANT le cutoff de selection.
- `context_recall`, `context_precision` : sur la sortie FINALE du selector
  (ce que le LLM generateur recoit reellement).

Bug corrige (nDCG > 1.0 observe avant ce chantier) : l'IDCG doit etre borne
par `min(k, |gold|)`, jamais `k` seul.

Format attendu de `eval/gold_retrieval.yaml` (fichier IMMUTABLE, lecture
seule) :
    - id: q1
      question: "..."
      type: multi_doc          # optionnel, informatif
      relevant_sources:
        - boot_sequence.md
        - long_procedures.md
      key_points:               # non utilise ici (cf. eval/generation_eval.py)
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

from src.domain.models import Answer

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_GOLD_PATH = _REPO_ROOT / "eval" / "gold_retrieval.yaml"
_DEFAULT_RESULTS_CSV_PATH = _REPO_ROOT / "eval" / "retrieval_eval_results.csv"

_NDCG_K = 10


@dataclass
class GoldItem:
    """Un item du gold set retrieval.

    Attributes:
        id: Identifiant court de la question (utilise dans les rapports).
        question: La question posee.
        relevant_sources: Sources jugees pertinentes pour y repondre (verite
            terrain COMPLETE : toute source hors de cette liste est un faux
            positif, sans exception).
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
    candidate_recall: float
    context_recall: float
    context_precision: float
    ndcg_at_10: float


def load_gold_set(path: Path = _DEFAULT_GOLD_PATH) -> list[GoldItem]:
    """Charge le gold set d'evaluation retrieval (fichier IMMUTABLE, lecture seule).

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


def to_sources(sources: list[str]) -> list[str]:
    """Deduplique une liste de sources (noms de fichiers), premier rang conserve.

    Projection obligatoire avant tout calcul de metrique : evite qu'un
    document decoupe en plusieurs chunks compte plusieurs fois (au numerateur
    comme au denominateur), et rend les differentes strategies de chunking
    comparables entre elles (le gold ne connait que des noms de fichiers).

    Args:
        sources: Sources (ex: `chunk.source`), dans un ordre quelconque -
            typiquement l'ordre de retrieval/reranking/selection.

    Returns:
        Les sources deduplicees, dans l'ordre de premiere apparition.
    """
    seen: set[str] = set()
    out: list[str] = []
    for source in sources:
        if source not in seen:
            seen.add(source)
            out.append(source)
    return out


def candidate_recall(retrieved_sources: list[str], gold: set[str]) -> float:
    """Garde-fou : proportion de sources pertinentes qui ont atteint le RETRIEVER
    (avant reranking). Ce n'est pas un score a optimiser : s'il est < 1.0, la
    source manquante n'a jamais eu la moindre chance d'etre selectionnee."""
    if not gold:
        return 0.0
    found = set(to_sources(retrieved_sources)) & gold
    return len(found) / len(gold)


def context_recall(selected_sources: list[str], gold: set[str]) -> float:
    """Proportion de sources pertinentes presentes dans la sortie FINALE du
    selector (le contexte reellement transmis au LLM generateur)."""
    if not gold:
        return 0.0
    selected = set(to_sources(selected_sources))
    return len(selected & gold) / len(gold)


def context_precision(selected_sources: list[str], gold: set[str]) -> float:
    """Proportion de la sortie FINALE du selector qui est effectivement
    pertinente. Contrairement a l'ancien `precision_at_k`, pas de plafond
    `|gold|/k` : `selected_sources` est de taille variable, 1.0 est atteignable."""
    selected = to_sources(selected_sources)
    if not selected:
        return 0.0
    hits = sum(1 for source in selected if source in gold)
    return hits / len(selected)


def ndcg_at_k(reranked_sources: list[str], gold: set[str], k: int = _NDCG_K) -> float:
    """nDCG@k sur l'ordre du RERANKER (avant le cutoff de selection), pertinence
    binaire par source deja dedupliquee (cf. `to_sources`).

    Bug corrige (observe avant ce chantier : nDCG > 1.0) : l'IDCG doit etre
    borne par `min(k, len(gold))`, jamais `k` seul - sinon le denominateur
    ideal est artificiellement trop petit dès que `len(gold) < k`.
    """
    ranked = to_sources(reranked_sources)
    dcg = sum(1.0 / math.log2(i + 2) for i, source in enumerate(ranked[:k]) if source in gold)
    ideal_hits = min(k, len(gold))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    score = dcg / idcg if idcg > 0 else 0.0
    assert 0.0 <= score <= 1.0 + 1e-9, f"ndcg_at_{k} hors bornes : {score}"
    return score


def evaluate_from_answer(answer: Answer, item: GoldItem, k: int = _NDCG_K) -> QueryMetrics:
    """Calcule les 4 metriques d'une question a partir de la reponse REELLE du
    pipeline (une seule execution : `Pipeline.run` expose deja
    `retrieved_sources`/`reranked_sources` dans `meta`, en plus des sources
    finales - cf. `src/domain/pipeline.py`).

    Args:
        answer: La reponse produite par `Pipeline.run(item.question)`.
        item: L'item du gold set correspondant (memes id/question).
        k: Rang de coupure pour `ndcg_at_10`.

    Returns:
        Les 4 metriques de retrieval pour cette question.
    """
    gold = set(item.relevant_sources)
    retrieved_sources = answer.meta.get("retrieved_sources", [])
    reranked_sources = answer.meta.get("reranked_sources", [])
    selected_sources = answer.meta.get("selected_sources", [])

    return QueryMetrics(
        id=item.id,
        question=item.question,
        candidate_recall=candidate_recall(retrieved_sources, gold),
        context_recall=context_recall(selected_sources, gold),
        context_precision=context_precision(selected_sources, gold),
        ndcg_at_10=ndcg_at_k(reranked_sources, gold, k=k),
    )


def evaluate(pipeline: Any, gold_set: list[GoldItem], k: int = _NDCG_K) -> list[QueryMetrics]:
    """Execute le pipeline REEL (retrieve -> rerank -> select -> generate) pour
    chaque question du gold set et calcule ses metriques de retrieval.

    Args:
        pipeline: Le `Pipeline` (domain/pipeline.py) deja construit/configure.
        gold_set: Les items du gold set (cf. `load_gold_set`).
        k: Rang de coupure applique a `ndcg_at_10`.

    Returns:
        Les metriques par question, dans l'ordre du gold set.
    """
    results = []
    for item in gold_set:
        answer = pipeline.run(item.question)
        results.append(evaluate_from_answer(answer, item, k=k))
    return results


def summarize(results: list[QueryMetrics]) -> dict[str, float]:
    """Moyenne les 4 metriques sur toutes les questions.

    Args:
        results: Les metriques par question (cf. `evaluate`).

    Returns:
        Un dict {candidate_recall, context_recall, context_precision, ndcg_at_10} moyenne.
    """
    n = len(results)
    if n == 0:
        return {
            "candidate_recall": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0,
            "ndcg_at_10": 0.0,
        }

    return {
        "candidate_recall": sum(r.candidate_recall for r in results) / n,
        "context_recall": sum(r.context_recall for r in results) / n,
        "context_precision": sum(r.context_precision for r in results) / n,
        "ndcg_at_10": sum(r.ndcg_at_10 for r in results) / n,
    }


def to_markdown_table(results: list[QueryMetrics], summary: dict[str, float]) -> str:
    """Formate les resultats (par question + moyenne) en tableau Markdown."""
    lines = [
        "| Question ID | Candidate Recall | Context Recall | Context Precision | nDCG@10 |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.id} | {r.candidate_recall:.3f} | {r.context_recall:.3f} | "
            f"{r.context_precision:.3f} | {r.ndcg_at_10:.3f} |"
        )
    lines.append(
        f"| **Moyenne** | **{summary['candidate_recall']:.3f}** | "
        f"**{summary['context_recall']:.3f}** | **{summary['context_precision']:.3f}** | "
        f"**{summary['ndcg_at_10']:.3f}** |"
    )
    return "\n".join(lines)


def write_csv(results: list[QueryMetrics], summary: dict[str, float], path: Path) -> None:
    """Ecrit les resultats (par question + moyenne) en CSV.

    Args:
        results: Les metriques par question.
        summary: Les metriques moyennes (cf. `summarize`).
        path: Fichier CSV de sortie (cree/ecrase).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["question_id", "candidate_recall", "context_recall", "context_precision", "ndcg_at_10"]
        )
        for r in results:
            writer.writerow(
                [r.id, r.candidate_recall, r.context_recall, r.context_precision, r.ndcg_at_10]
            )
        writer.writerow(
            [
                "MEAN",
                summary["candidate_recall"],
                summary["context_recall"],
                summary["context_precision"],
                summary["ndcg_at_10"],
            ]
        )


def main() -> None:
    """Charge la config + le gold set reels, evalue le pipeline complet, ecrit les rapports."""
    from pathlib import Path as _Path

    from src.application.answer import build_pipeline
    from src.cli import load_chunks, load_config, load_system_prompt

    config: dict[str, Any] = load_config()

    chunks_path = _Path(config["vectorstore"]["chunks_path"])
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"index introuvable ({chunks_path}) : executer "
            "`uv run python -m src.cli index` avant l'evaluation"
        )
    chunks = load_chunks(chunks_path)
    system_prompt = load_system_prompt()
    pipeline = build_pipeline(config, chunks, system_prompt)

    gold_set = load_gold_set()
    k = config["pipeline"].get("ndcg_k", _NDCG_K)

    results = evaluate(pipeline, gold_set, k=k)
    summary = summarize(results)

    print(to_markdown_table(results, summary))

    write_csv(results, summary, _DEFAULT_RESULTS_CSV_PATH)
    print(f"\nResultats CSV ecrits dans {_DEFAULT_RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
