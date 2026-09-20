"""Etude d'ablation : compare chaque config/experiments/*.yaml au baseline
(config/default.yaml) sur retrieval + generation, sur les 5 questions du gold
set, et ecrit un tableau comparatif Markdown + CSV trie.

Chaque fichier config/experiments/*.yaml ne change QU'UNE variable (une seule
dimension - parfois 2 cles liees quand c'est necessaire pour isoler cette
dimension, ex: retrieval.hybrid + retrieval.strategy pour isoler le mode
"bm25 seul", documente dans le fichier concerne) par rapport a
config/default.yaml. Un point d'entree unique (`main`) execute TOUTES les
combinaisons une-variable-a-la-fois trouvees dans config/experiments/,
couvrant les 6 dimensions du projet : chunking, embedding, retrieval,
reranking, selection, llm.

(Re)indexation : seuls les experiences qui changent `chunking.*` ou
`embedding.*` necessitent un nouvel index (les vecteurs/chunks dependent de
ces deux dimensions) ; chacune obtient son propre index isole
(data/ablation/<nom>/), construit une seule fois puis reutilise (cache) aux
executions suivantes. Les autres experiences (retrieval, reranking,
selection, llm) reutilisent l'index baseline (construit une seule fois si
absent).

Reproductibilite : `random.seed(SEED)` est fixe, et chaque config utilise sa
propre temperature LLM deja basse (0.1 par defaut). Limite assumee : les
appels a des API distantes (Ollama, Claude) ne sont pas garantis bit-a-bit
identiques d'une execution a l'autre (aucun controle direct sur leur RNG
interne) - la reproductibilite ici porte sur TOUT ce que le projet controle
(chunking, retrieval, reranking, selection, choix des questions).

Usage:
    uv run python -m eval.run_ablation
"""

import copy
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

SEED = 42

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _REPO_ROOT / "config" / "experiments"
_ABLATION_DATA_DIR = _REPO_ROOT / "data" / "ablation"
_DEFAULT_RESULTS_CSV_PATH = _REPO_ROOT / "eval" / "run_ablation_results.csv"
_CHECKPOINT_PATH = _REPO_ROOT / "eval" / "run_ablation_checkpoint.json"


def _load_checkpoint(path: Path = _CHECKPOINT_PATH) -> dict[str, "ExperimentResult"]:
    """Recharge les experiences deja terminees (relance sans les refaire)."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {name: ExperimentResult(**item) for name, item in data.items()}


def _save_checkpoint(checkpoint: dict[str, "ExperimentResult"], path: Path = _CHECKPOINT_PATH) -> None:
    """Sauvegarde apres CHAQUE experience (relance = reprise, pas redemarrage)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: asdict(r) for name, r in checkpoint.items()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

_REINDEX_SECTIONS = {"chunking", "embedding"}


@dataclass
class ExperimentResult:
    """Resultat agrege (retrieval + generation) pour une config."""

    name: str
    changed_keys: list[str]
    retrieval_metrics: dict[str, float]
    generation_metrics: dict[str, float]

    @property
    def composite_score(self) -> float:
        """Moyenne non ponderee de toutes les metriques (retrieval + generation),
        toutes deja sur l'echelle [0, 1]. Sert uniquement a trier le tableau."""
        values = list(self.retrieval_metrics.values()) + list(self.generation_metrics.values())
        return sum(values) / len(values) if values else 0.0


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Fusionne recursivement `override` dans une copie de `base`.

    Contrairement a un merge superficiel (`{**base, **override}`), preserve
    les cles soeurs non modifiees d'une sous-section (ex: fusionner
    {"llm": {"model": "x"}} dans {"llm": {"model": "y", "temperature": 0.1}}
    donne {"llm": {"model": "x", "temperature": 0.1}}, pas la perte de
    `temperature`).

    Args:
        base: Configuration de base (non modifiee).
        override: Valeurs a surcharger (peut etre partiel, imbrique).

    Returns:
        Une nouvelle configuration fusionnee.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def changed_keys(override: dict[str, Any], prefix: str = "") -> list[str]:
    """Liste les cles modifiees par `override`, sous forme de chemins pointes.

    Args:
        override: Le dict d'override (cf. `deep_merge`).
        prefix: Prefixe interne pour la recursion (laisser vide en appel externe).

    Returns:
        Les chemins pointes (ex: ["llm.model"]), tries.
    """
    paths: list[str] = []
    for key, value in override.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            paths.extend(changed_keys(value, prefix=path))
        else:
            paths.append(path)
    return sorted(paths)


def requires_reindex(override: dict[str, Any]) -> bool:
    """True si `override` touche `chunking.*` ou `embedding.*` (vectorisation
    dependante), donc necessite un nouvel index."""
    return bool(_REINDEX_SECTIONS & override.keys())


def discover_experiment_configs(directory: Path = _EXPERIMENTS_DIR) -> list[Path]:
    """Liste tous les fichiers config/experiments/*.yaml, tries par nom."""
    return sorted(directory.glob("*.yaml"))


def load_experiment_override(path: Path) -> dict[str, Any]:
    """Charge un fichier d'override d'experience (dict partiel, imbrique)."""
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def ensure_index(
    config: dict[str, Any], experiment_name: str, needs_reindex: bool
) -> tuple[dict[str, Any], list[Any]]:
    """Garantit qu'un index existe pour `config`, en construit un si besoin.

    Args:
        config: Configuration complete (deja fusionnee).
        experiment_name: Nom de l'experience (utilise pour isoler son index
            si `needs_reindex`, ex: "data/ablation/<nom>/").
        needs_reindex: Si True, utilise un index isole a cette experience
            (construit une seule fois, ensuite reutilise) ; sinon reutilise
            l'index baseline de `config` tel quel (construit si absent).

    Returns:
        (config effectivement utilise pour l'indexation, chunks indexes).
    """
    from src.cli import load_chunks, run_index

    effective_config = config
    if needs_reindex:
        exp_dir = _ABLATION_DATA_DIR / experiment_name
        effective_config = deep_merge(
            config,
            {
                "vectorstore": {
                    "persist_directory": str(exp_dir / "chroma"),
                    "chunks_path": str(exp_dir / "chunks.json"),
                }
            },
        )

    chunks_path = Path(effective_config["vectorstore"]["chunks_path"])
    if chunks_path.exists():
        return effective_config, load_chunks(chunks_path)

    chunks = run_index(effective_config)
    return effective_config, chunks


def run_experiment(
    name: str,
    override: dict[str, Any],
    base_config: dict[str, Any],
    gold_retrieval_items: list[Any],
    gold_generation_cases: list[Any],
    system_prompt: str,
    judge_llm_raw: Any,
    ragas_llm: Any,
    ragas_embeddings: Any,
) -> ExperimentResult:
    """Execute retrieval + generation pour une config et agrege ses metriques.

    Args:
        name: Nom de l'experience (nom de fichier sans extension, ou
            "baseline").
        override: Le dict d'override (vide pour le baseline).
        base_config: La config par defaut complete (config/default.yaml).
        gold_retrieval_items: Gold set retrieval (cf. `eval.retrieval_eval.load_gold_set`).
        gold_generation_cases: Cas de generation positifs (cf.
            `eval.generation_eval.load_gold_cases`), memes questions.
        system_prompt: Contenu du system prompt.
        judge_llm_raw: LLM juge Claude brut (cf. `eval.generation_eval.load_judge_llm`).
        ragas_llm: LLM juge deja enveloppe pour RAGAS.
        ragas_embeddings: Embedder local deja enveloppe pour RAGAS.

    Returns:
        Le `ExperimentResult` agrege pour cette experience.
    """
    from src.adapters.embedding.factory import create_embedder
    from src.adapters.retrieval.factory import create_retriever
    from src.adapters.vectorstore.chroma import create_vectorstore
    from eval.generation_eval import custom_judge_eval, ragas_eval, run_pipeline_on_cases
    from eval.retrieval_eval import evaluate as evaluate_retrieval
    from eval.retrieval_eval import summarize as summarize_retrieval
    from src.application.answer import build_pipeline

    config = deep_merge(base_config, override)
    needs_reindex = requires_reindex(override)
    config, chunks = ensure_index(config, experiment_name=name, needs_reindex=needs_reindex)

    embedder = create_embedder(config["embedding"])
    vectorstore = create_vectorstore(config["vectorstore"])
    retriever = create_retriever(config["retrieval"], embedder, vectorstore, chunks)

    retrieval_results = evaluate_retrieval(retriever, gold_retrieval_items, k=config["pipeline"]["retrieve_k"])
    retrieval_metrics = summarize_retrieval(retrieval_results)

    pipeline = build_pipeline(config, chunks, system_prompt)
    generation_results = run_pipeline_on_cases(pipeline, gold_generation_cases)

    custom_scores = custom_judge_eval(judge_llm_raw, generation_results)
    ragas_scores = ragas_eval(generation_results, ragas_llm, ragas_embeddings)

    n = len(generation_results)
    generation_metrics = {
        "faithfulness_custom": sum(s.faithfulness for s in custom_scores.values()) / n,
        "relevancy_custom": sum(s.relevancy for s in custom_scores.values()) / n,
        "faithfulness_ragas": sum(v["faithfulness"] for v in ragas_scores.values()) / n,
        "answer_relevancy_ragas": sum(v["answer_relevancy"] for v in ragas_scores.values()) / n,
        "context_precision_ragas": sum(v["context_precision"] for v in ragas_scores.values()) / n,
        "context_recall_ragas": sum(v["context_recall"] for v in ragas_scores.values()) / n,
    }

    return ExperimentResult(
        name=name,
        changed_keys=changed_keys(override),
        retrieval_metrics=retrieval_metrics,
        generation_metrics=generation_metrics,
    )


def to_markdown_table(results: list[ExperimentResult]) -> str:
    """Formate les resultats en tableau Markdown, trie par score composite decroissant."""
    ordered = sorted(results, key=lambda r: r.composite_score, reverse=True)

    lines = [
        "| Experience | Variable(s) changee(s) | Precision@k | Recall@k | MRR | nDCG@k | "
        "Faith. (custom) | Relev. (custom) | Faith. (RAGAS) | Ans. Relev. | Ctx Prec. | Ctx Recall | Score |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in ordered:
        rm, gm = r.retrieval_metrics, r.generation_metrics
        changed = ", ".join(r.changed_keys) if r.changed_keys else "(baseline)"
        lines.append(
            f"| {r.name} | {changed} | "
            f"{rm['precision_at_k']:.3f} | {rm['recall_at_k']:.3f} | {rm['mrr']:.3f} | {rm['ndcg_at_k']:.3f} | "
            f"{gm['faithfulness_custom']:.3f} | {gm['relevancy_custom']:.3f} | "
            f"{gm['faithfulness_ragas']:.3f} | {gm['answer_relevancy_ragas']:.3f} | "
            f"{gm['context_precision_ragas']:.3f} | {gm['context_recall_ragas']:.3f} | "
            f"**{r.composite_score:.3f}** |"
        )
    return "\n".join(lines)


def write_csv(results: list[ExperimentResult], path: Path) -> None:
    """Ecrit les resultats (tries par score composite decroissant) en CSV."""
    ordered = sorted(results, key=lambda r: r.composite_score, reverse=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "changed_keys",
                "precision_at_k",
                "recall_at_k",
                "mrr",
                "ndcg_at_k",
                "faithfulness_custom",
                "relevancy_custom",
                "faithfulness_ragas",
                "answer_relevancy_ragas",
                "context_precision_ragas",
                "context_recall_ragas",
                "composite_score",
            ]
        )
        for r in ordered:
            rm, gm = r.retrieval_metrics, r.generation_metrics
            writer.writerow(
                [
                    r.name,
                    ";".join(r.changed_keys),
                    rm["precision_at_k"],
                    rm["recall_at_k"],
                    rm["mrr"],
                    rm["ndcg_at_k"],
                    gm["faithfulness_custom"],
                    gm["relevancy_custom"],
                    gm["faithfulness_ragas"],
                    gm["answer_relevancy_ragas"],
                    gm["context_precision_ragas"],
                    gm["context_recall_ragas"],
                    r.composite_score,
                ]
            )


def main() -> None:
    """Point d'entree unique : baseline + toutes les config/experiments/*.yaml."""
    from eval.generation_eval import load_gold_cases, load_ragas_judge_and_embeddings, load_judge_llm
    from eval.retrieval_eval import load_gold_set
    from src.cli import load_config, load_system_prompt

    random.seed(SEED)

    base_config = load_config()
    system_prompt = load_system_prompt()
    gold_retrieval_items = load_gold_set()
    gold_generation_cases = load_gold_cases()

    judge_llm_raw = load_judge_llm()
    ragas_llm, ragas_embeddings = load_ragas_judge_and_embeddings(base_config["embedding"])

    experiments: list[tuple[str, dict[str, Any]]] = [("baseline", {})]
    for path in discover_experiment_configs():
        experiments.append((path.stem, load_experiment_override(path)))

    checkpoint = _load_checkpoint()
    if checkpoint:
        print(f"Checkpoint trouve : {len(checkpoint)} experience(s) deja faite(s), reprise.")

    for name, override in experiments:
        if name in checkpoint:
            print(f"--- {name} : deja fait (checkpoint), ignore ---")
            continue
        print(f"--- {name} ({', '.join(changed_keys(override)) or 'baseline'}) ---")
        checkpoint[name] = run_experiment(
            name=name,
            override=override,
            base_config=base_config,
            gold_retrieval_items=gold_retrieval_items,
            gold_generation_cases=gold_generation_cases,
            system_prompt=system_prompt,
            judge_llm_raw=judge_llm_raw,
            ragas_llm=ragas_llm,
            ragas_embeddings=ragas_embeddings,
        )
        _save_checkpoint(checkpoint)  # sauvegarde immediate : jamais a refaire

    results = [checkpoint[name] for name, _ in experiments]
    print("\n" + to_markdown_table(results))

    write_csv(results, _DEFAULT_RESULTS_CSV_PATH)
    print(f"\nResultats CSV ecrits dans {_DEFAULT_RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
