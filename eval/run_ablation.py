"""Etude d'ablation : compare chaque config/experiments/*.yaml au baseline
(config/default.yaml), sur 4 metriques de retrieval + 2 metriques de
generation, sur les 5 questions du gold set (chantier "relative_threshold",
etapes 3+4+5).

Determinisme (etape 4) :
- Chaque configuration est executee `N_REPETITIONS` fois (3 par defaut) ; le
  CSV reporte `<metrique>_mean` et `<metrique>_std`. Les metriques de
  retrieval (candidate_recall, context_recall, context_precision, ndcg_at_10)
  sont deterministes (aucun echantillonnage dans retrieve/rerank/select) et
  ont donc un std structurellement nul ; seules `faithfulness` et
  `answer_correctness` varient reellement d'une repetition a l'autre
  (echantillonnage du generateur Ollama, cf. `llm.temperature`).
- `main()` marque NON SIGNIFICATIF (dans le rapport, pas dans ce CSV) tout
  ecart entre deux runs inferieur a `_SIGNIFICANCE_STD_MULTIPLIER * std`.
- Le runner REFUSE de demarrer si une experience modifie plus d'UNE SECTION
  de config de premier niveau (cf. `_refuse_if_multi_section` : interpretation
  retenue et documentee pour la regle "plus d'une cle", ambigue en l'etat -
  `retrieval_bm25_only.yaml` touche 2 chemins (retrieval.hybrid,
  retrieval.strategy) mais une seule SECTION ("retrieval"), necessaires
  ensemble pour isoler une seule variable, deja documente dans ce fichier).
- Le juge LLM (cf. `eval.generation_eval.load_judge_llm`) est FIXE pour toute
  la duree de l'ablation, y compris pour les experiences qui font varier
  `llm.model` (le generateur evalue) : jamais le meme modele des deux cotes.
  Logue via `judge_model` dans chaque ligne du CSV.
- `config_hash` (hash de la config effective fusionnee) et `reranker_version`
  (nom du modele de reranking) sont aussi logues par ligne.
- Pre-check obligatoire : `eval.validate_gold.validate_gold()` avant tout run.

(Re)indexation : seuls les experiences qui changent `chunking.*` ou
`embedding.*` necessitent un nouvel index, isole (data/ablation/<nom>/),
construit une seule fois puis reutilise. Les autres reutilisent l'index
baseline (construit une seule fois si absent).

Cout reel : chaque question judgee coute ~1 appel juge pour l'extraction des
affirmations + 1 appel/affirmation (faithfulness) + 1 appel/key_point
(answer_correctness) - de l'ordre de 5-10 appels juge par question. Sur 5
questions x N_REPETITIONS x (1 baseline + len(config/experiments/*.yaml)),
le nombre total d'appels juge reels est substantiel : verifie/confirme avec
l'utilisateur avant un lancement complet (cf. rapport de ce chantier).

Usage:
    uv run python -m eval.run_ablation
"""

import copy
import csv
import hashlib
import json
import random
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

SEED = 42
N_REPETITIONS = 3
_SIGNIFICANCE_STD_MULTIPLIER = 2
_MIN_VALID_CONTEXT_RECALL = 0.98
_NDCG_K = 10

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _REPO_ROOT / "config" / "experiments"
_ABLATION_DATA_DIR = _REPO_ROOT / "data" / "ablation"
_DEFAULT_RESULTS_CSV_PATH = _REPO_ROOT / "eval" / "results_v2.csv"
_CHECKPOINT_PATH = _REPO_ROOT / "eval" / "run_ablation_checkpoint_v2.json"

_REINDEX_SECTIONS = {"chunking", "embedding"}

_RETRIEVAL_METRIC_NAMES = ("candidate_recall", "context_recall", "context_precision", "ndcg_at_10")


@dataclass
class RepetitionResult:
    """Metriques d'UNE repetition (moyennees sur les 5 questions du gold set)."""

    retrieval_metrics: dict[str, float]
    faithfulness: float | None
    n_claims_total: int
    answer_correctness: float | None
    n_key_points_covered: int
    n_key_points_total: int
    uncovered_key_point_ids: list[str]


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Moyenne et ecart-type (0.0 si un seul point ou tous identiques)."""
    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


@dataclass
class ExperimentResult:
    """Resultat agrege (sur les repetitions) pour une config."""

    name: str
    changed_keys: list[str]
    config_hash: str
    judge_model: str
    reranker_version: str
    repetitions: list[RepetitionResult] = field(default_factory=list)

    def metric_mean_std(self, metric: str) -> tuple[float, float]:
        """(mean, std) d'une metrique sur les repetitions.

        Les repetitions ou la metrique est None (ex: `faithfulness` pour un
        run 100% abstention, ou `answer_correctness` sans key_points) sont
        ignorees dans le calcul ; retourne (0.0, 0.0) si aucune repetition
        n'a de valeur definie.
        """
        if metric in _RETRIEVAL_METRIC_NAMES:
            values = [r.retrieval_metrics[metric] for r in self.repetitions]
        elif metric == "faithfulness":
            values = [r.faithfulness for r in self.repetitions if r.faithfulness is not None]
        elif metric == "answer_correctness":
            values = [r.answer_correctness for r in self.repetitions if r.answer_correctness is not None]
        else:
            raise ValueError(f"metrique inconnue : {metric!r}")
        return _mean_std([v for v in values if v is not None])

    @property
    def composite(self) -> float | None:
        """Score composite pondere, ou None si le run est INVALIDE (garde-fou).

        Garde-fou obligatoire (etape 3.3) : `context_recall_mean < 0.98`
        rend le run invalide (None), jamais un score bas - une moyenne
        ponderee simple permettrait a un rappel catastrophique de compenser
        ailleurs (bug reel constate : `reranking_disabled`, context_recall
        RAGAS = 0.13 dans l'ancien harness, finissait pourtant devant
        `llm_smollm` au score composite).
        `faithfulness_mean` == None (abstention totale) -> composite = 0.0.
        """
        context_recall_mean, _ = self.metric_mean_std("context_recall")
        if context_recall_mean < _MIN_VALID_CONTEXT_RECALL:
            return None

        faithfulness_mean, _ = self.metric_mean_std("faithfulness")
        has_faithfulness = any(r.faithfulness is not None for r in self.repetitions)
        if not has_faithfulness:
            return 0.0

        context_precision_mean, _ = self.metric_mean_std("context_precision")
        ndcg_mean, _ = self.metric_mean_std("ndcg_at_10")
        answer_correctness_mean, _ = self.metric_mean_std("answer_correctness")

        return (
            0.30 * context_precision_mean
            + 0.15 * ndcg_mean
            + 0.30 * faithfulness_mean
            + 0.25 * answer_correctness_mean
        )

    @property
    def all_uncovered_key_point_ids(self) -> list[str]:
        """Union (dedupliquee, ordre stable) des key_points non couverts sur
        toutes les repetitions - signal le plus actionnable du rapport."""
        seen: set[str] = set()
        out: list[str] = []
        for rep in self.repetitions:
            for kp_id in rep.uncovered_key_point_ids:
                if kp_id not in seen:
                    seen.add(kp_id)
                    out.append(kp_id)
        return out


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


def refuse_if_multi_section(name: str, override: dict[str, Any]) -> None:
    """Refuse une experience qui touche plus d'UNE section de config de
    premier niveau (etape 4 : attribution non ambigue du gain/de la perte a
    une seule variable).

    Args:
        name: Nom de l'experience (pour le message d'erreur).
        override: Le dict d'override de cette experience.

    Raises:
        ValueError: Si `override` touche plus d'une section de premier niveau.
    """
    sections = set(override.keys())
    if len(sections) > 1:
        raise ValueError(
            f"experience {name!r} refusee : modifie {len(sections)} sections de config "
            f"({', '.join(sorted(sections))}) - une experience doit isoler UNE seule "
            "variable (une seule section de premier niveau), sinon le gain/la perte "
            "observe n'est pas attribuable sans ambiguite."
        )


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


def config_hash(config: dict[str, Any]) -> str:
    """Hash court (12 hex) de la config effective, pour tracabilite (etape 4)."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def gold_hash(gold_path: Path) -> str:
    """Hash court (12 hex) du contenu brut du gold set, pour tracabilite (etape 4)."""
    return hashlib.sha256(gold_path.read_bytes()).hexdigest()[:12]


def _load_checkpoint(path: Path = _CHECKPOINT_PATH) -> dict[str, ExperimentResult]:
    """Recharge les experiences deja terminees (relance sans les refaire)."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    results: dict[str, ExperimentResult] = {}
    for name, item in data.items():
        reps = [RepetitionResult(**rep) for rep in item.pop("repetitions")]
        results[name] = ExperimentResult(repetitions=reps, **item)
    return results


def _save_checkpoint(checkpoint: dict[str, ExperimentResult], path: Path = _CHECKPOINT_PATH) -> None:
    """Sauvegarde apres CHAQUE experience (relance = reprise, pas redemarrage)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: asdict(r) for name, r in checkpoint.items()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def run_repetition(
    pipeline: Any,
    gold_retrieval_items: list[Any],
    gold_generation_cases: list[Any],
    judge: Any,
) -> RepetitionResult:
    """Execute UNE repetition (les 5 questions du gold, retrieval + generation
    + jugement), et agrege les metriques (moyenne sur les questions).

    Args:
        pipeline: Le `Pipeline` (domain/pipeline.py) deja construit.
        gold_retrieval_items: Gold set retrieval (cf. `eval.retrieval_eval.load_gold_set`).
        gold_generation_cases: Cas de generation positifs (cf.
            `eval.generation_eval.load_gold_cases`), memes questions.
        judge: Le LLM juge deja construit (cf. `eval.generation_eval.load_judge_llm`).

    Returns:
        Le `RepetitionResult` agrege sur les questions positives.

    Note : `gold_retrieval_items` et `gold_generation_cases` doivent
    correspondre 1:1 par position (memes questions, meme ordre) - le pipeline
    n'est execute qu'UNE SEULE FOIS par question (jamais deux), les metriques
    de retrieval ET de generation derivent de la MEME `Answer`.
    """
    from eval.generation_eval import answer_correctness, faithfulness
    from eval.retrieval_eval import evaluate_from_answer, summarize as summarize_retrieval

    retrieval_metrics_per_q = []
    faithfulness_scores: list[float] = []
    n_claims_total = 0
    correctness_scores: list[float] = []
    n_covered_total = 0
    n_total_total = 0
    uncovered_ids: list[str] = []

    for retrieval_item, generation_case in zip(gold_retrieval_items, gold_generation_cases):
        answer = pipeline.run(retrieval_item.question)

        retrieval_metrics_per_q.append(evaluate_from_answer(answer, retrieval_item, k=_NDCG_K))

        context = "\n\n".join(answer.meta.get("selected_chunk_texts", []))
        faith = faithfulness(answer.text, context, judge)
        n_claims_total += faith.n_claims
        if faith.score is not None:
            faithfulness_scores.append(faith.score)

        correctness = answer_correctness(
            generation_case.id, generation_case.key_points, answer.text, judge
        )
        n_covered_total += correctness.n_covered
        n_total_total += correctness.n_total
        uncovered_ids.extend(correctness.uncovered_key_point_ids)
        if correctness.score is not None:
            correctness_scores.append(correctness.score)

    retrieval_metrics = summarize_retrieval(retrieval_metrics_per_q)

    return RepetitionResult(
        retrieval_metrics=retrieval_metrics,
        faithfulness=(sum(faithfulness_scores) / len(faithfulness_scores)) if faithfulness_scores else None,
        n_claims_total=n_claims_total,
        answer_correctness=(sum(correctness_scores) / len(correctness_scores)) if correctness_scores else None,
        n_key_points_covered=n_covered_total,
        n_key_points_total=n_total_total,
        uncovered_key_point_ids=uncovered_ids,
    )


def run_experiment(
    name: str,
    override: dict[str, Any],
    base_config: dict[str, Any],
    gold_retrieval_items: list[Any],
    gold_generation_cases: list[Any],
    system_prompt: str,
    judge: Any,
    judge_model: str,
    n_repetitions: int = N_REPETITIONS,
) -> ExperimentResult:
    """Execute `n_repetitions` fois retrieval + generation pour une config et
    agrege ses metriques (moyenne/ecart-type sur les repetitions).

    Args:
        name: Nom de l'experience (nom de fichier sans extension, ou "baseline").
        override: Le dict d'override (vide pour le baseline).
        base_config: La config par defaut complete (config/default.yaml).
        gold_retrieval_items: Gold set retrieval.
        gold_generation_cases: Cas de generation positifs, memes questions.
        system_prompt: Contenu du system prompt.
        judge: Le LLM juge deja construit (fixe pour toute l'ablation).
        judge_model: Nom du modele juge (logue tel quel, pour tracabilite).
        n_repetitions: Nombre de repetitions (determinisme, etape 4).

    Returns:
        Le `ExperimentResult` agrege pour cette experience.
    """
    from src.adapters.embedding.factory import create_embedder
    from src.adapters.retrieval.factory import create_retriever
    from src.adapters.reranking.factory import create_reranker
    from src.adapters.vectorstore.chroma import create_vectorstore
    from src.application.answer import build_pipeline

    refuse_if_multi_section(name, override)

    config = deep_merge(base_config, override)
    needs_reindex = requires_reindex(override)
    config, chunks = ensure_index(config, experiment_name=name, needs_reindex=needs_reindex)

    pipeline = build_pipeline(config, chunks, system_prompt)

    repetitions = [
        run_repetition(pipeline, gold_retrieval_items, gold_generation_cases, judge)
        for _ in range(n_repetitions)
    ]

    return ExperimentResult(
        name=name,
        changed_keys=changed_keys(override),
        config_hash=config_hash(config),
        judge_model=judge_model,
        reranker_version=config["reranking"].get("model_name", "(disabled)"),
        repetitions=repetitions,
    )


def to_markdown_table(results: list[ExperimentResult]) -> str:
    """Formate les resultats en tableau Markdown, trie par composite decroissant
    (les runs INVALIDES - composite None - en dernier)."""
    ordered = sorted(
        results, key=lambda r: (r.composite is None, -(r.composite or 0.0))
    )

    lines = [
        "| Experience | Variable changee | Candidate Recall | Context Recall | Context Precision | "
        "nDCG@10 | Faithfulness | Answer Correctness | Composite |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in ordered:
        changed = ", ".join(r.changed_keys) if r.changed_keys else "(baseline)"
        cand_r, _ = r.metric_mean_std("candidate_recall")
        ctx_r, _ = r.metric_mean_std("context_recall")
        ctx_p, _ = r.metric_mean_std("context_precision")
        ndcg, _ = r.metric_mean_std("ndcg_at_10")
        faith, _ = r.metric_mean_std("faithfulness")
        correct, _ = r.metric_mean_std("answer_correctness")
        composite = "INVALIDE (recall<0.98)" if r.composite is None else f"**{r.composite:.3f}**"
        lines.append(
            f"| {r.name} | {changed} | {cand_r:.3f} | {ctx_r:.3f} | {ctx_p:.3f} | "
            f"{ndcg:.3f} | {faith:.3f} | {correct:.3f} | {composite} |"
        )
    return "\n".join(lines)


def write_csv(results: list[ExperimentResult], path: Path) -> None:
    """Ecrit les resultats (tries par composite decroissant, invalides en
    dernier) en CSV, avec `_mean`/`_std` pour chaque metrique numerique."""
    ordered = sorted(
        results, key=lambda r: (r.composite is None, -(r.composite or 0.0))
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "changed_key",
                "candidate_recall_mean", "candidate_recall_std",
                "context_recall_mean", "context_recall_std",
                "context_precision_mean", "context_precision_std",
                "ndcg_at_10_mean", "ndcg_at_10_std",
                "faithfulness_mean", "faithfulness_std",
                "answer_correctness_mean", "answer_correctness_std",
                "n_claims_total",
                "n_key_points_covered",
                "n_key_points_total",
                "composite",
                "config_hash",
                "judge_model",
            ]
        )
        for r in ordered:
            row: list[Any] = [r.name, ";".join(r.changed_keys) or "(baseline)"]
            for metric in (*_RETRIEVAL_METRIC_NAMES, "faithfulness", "answer_correctness"):
                mean, std = r.metric_mean_std(metric)
                row.extend([mean, std])
            # n_claims_total / n_key_points_* : sommes sur les repetitions
            # (pas des metriques [0,1] moyennables comme celles ci-dessus).
            row.append(sum(rep.n_claims_total for rep in r.repetitions))
            row.append(sum(rep.n_key_points_covered for rep in r.repetitions))
            row.append(sum(rep.n_key_points_total for rep in r.repetitions))
            row.append("" if r.composite is None else r.composite)
            row.append(r.config_hash)
            row.append(r.judge_model)
            writer.writerow(row)


def main() -> None:
    """Point d'entree unique : baseline + toutes les config/experiments/*.yaml."""
    from eval.generation_eval import load_gold_cases, load_judge_llm, judge_model_name, negative_cases
    from eval.retrieval_eval import load_gold_set
    from eval.validate_gold import validate_gold
    from src.cli import load_config, load_system_prompt

    validate_gold()  # pre-check obligatoire (etape 3.4) : arrete tout si le gold est incoherent

    random.seed(SEED)

    base_config = load_config()
    system_prompt = load_system_prompt()
    gold_retrieval_items = load_gold_set()
    gold_generation_cases = load_gold_cases()

    judge = load_judge_llm()
    judge_model = judge_model_name()

    experiments: list[tuple[str, dict[str, Any]]] = [("baseline", {})]
    for path in discover_experiment_configs():
        experiments.append((path.stem, load_experiment_override(path)))

    for name, override in experiments:
        refuse_if_multi_section(name, override)  # echoue tot, avant tout appel reel

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
            judge=judge,
            judge_model=judge_model,
        )
        _save_checkpoint(checkpoint)  # sauvegarde immediate : jamais a refaire

    results = [checkpoint[name] for name, _ in experiments]
    print("\n" + to_markdown_table(results))

    write_csv(results, _DEFAULT_RESULTS_CSV_PATH)
    print(f"\nResultats CSV ecrits dans {_DEFAULT_RESULTS_CSV_PATH}")
    print(f"gold_hash={gold_hash(_REPO_ROOT / 'eval' / 'gold_retrieval.yaml')}")


if __name__ == "__main__":
    main()
