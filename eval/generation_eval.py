"""Evalue la generation du pipeline RAG via DEUX metriques orthogonales,
calculees par un juge LLM local (Claude ou Bedrock, jamais le petit modele
1.7-2B teste comme generateur), un appel PAR AFFIRMATION - jamais un appel
global. Aucune dependance externe de scoring (pas de sacrebleu, rouge, nltk,
bert-score, ragas) : chantier "relative_threshold", etape 3.2.

- `faithfulness` (anti-hallucination) : le modele invente-t-il ? Extraction
  des affirmations atomiques de la reponse, puis verification de chacune
  contre le CONTEXTE REELLEMENT FOURNI au generateur (jamais le gold, jamais
  le corpus entier).
- `answer_correctness` (anti-omission) : dit-il ce qu'il fallait dire ?
  Couverture de chaque `key_point` du gold set (verite terrain IMMUTABLE,
  jamais modifiee) par la reponse generee.

Les deux sont necessaires ensemble : un modele peut saturer `faithfulness`
en ne disant presque rien (`answer_correctness` l'en empeche) - mode de
defaillance reel observe (`llm_smollm`, faithfulness_custom=1.00 avec
relevancy=0.08 dans l'ancien harness).

Le fournisseur du juge est choisi par la variable d'environnement
LLM_PROVIDER (.env, jamais committee) :
- "anthropic" (defaut, retro-compatible) : API Anthropic directe via
  langchain-anthropic (`ChatAnthropic`), modele configurable via
  ANTHROPIC_JUDGE_MODEL, cle lue depuis ANTHROPIC_API_KEY.
- "bedrock" : Amazon Bedrock (API Converse) via langchain-aws
  (`ChatBedrockConverse`), modele lu depuis BEDROCK_JUDGE_MODEL_ID, region
  depuis AWS_REGION. Authentification par jeton porteur
  (AWS_BEARER_TOKEN_BEDROCK) : ce code ne lit QUE sa presence, jamais sa
  valeur - botocore le detecte lui-meme dans l'environnement du processus,
  qui n'est donc jamais logue ni transmis explicitement en Python.

Determinisme (chantier "relative_threshold", etape 4) : `temperature=0` sur
les deux fournisseurs - le maximum de determinisme expose par leurs API
respectives. `top_p` deliberement NON transmis en plus : verifie en direct,
le modele Bedrock cible (Claude Sonnet 4.6 via Converse) rejette une requete
specifiant temperature ET top_p simultanement ("cannot both be specified for
this model") - erreur API reelle, pas une hypothese. `top_p` est de toute
facon sans effet des lors que `temperature=0` force un decodage quasi-glouton.
Limite verifiee et documentee : ni `ChatAnthropic` ni `ChatBedrockConverse`
n'exposent de parametre `seed` (contrairement a certaines API type OpenAI) -
la reproductibilite bit-a-bit du juge n'est donc PAS garantie meme a
temperature=0, seulement rendue la plus stable possible.

Usage:
    uv run python -m eval.generation_eval
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.domain.models import Answer

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_GOLD_PATH = _REPO_ROOT / "eval" / "gold_retrieval.yaml"
_DEFAULT_RESULTS_CSV_PATH = _REPO_ROOT / "eval" / "generation_eval_results.csv"

_JUDGE_PROVIDER_ENV_VAR = "LLM_PROVIDER"

_JUDGE_API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
_JUDGE_MODEL_ENV_VAR = "ANTHROPIC_JUDGE_MODEL"

_BEDROCK_BEARER_TOKEN_ENV_VAR = "AWS_BEARER_TOKEN_BEDROCK"
_BEDROCK_REGION_ENV_VAR = "AWS_REGION"
_BEDROCK_MODEL_ID_ENV_VAR = "BEDROCK_JUDGE_MODEL_ID"

_JUDGE_TEMPERATURE = 0.0

# Questions hors-corpus : le pipeline doit s'abstenir sur chacune d'elles.
# Aucun key_point (rien a couvrir) : seule `faithfulness` s'applique.
NEGATIVE_QUESTIONS = [
    "What is the capital of France?",
    "Who won the FIFA World Cup in 2022?",
]

_CLAIM_EXTRACTION_PROMPT = """Découpe la réponse ci-dessous en affirmations atomiques et vérifiables.
Une affirmation = un seul fait. Ignore les formules de politesse, les
transitions et les reformulations de la question.

RÉPONSE :
{answer}

Retourne une affirmation par ligne, sans numérotation, sans commentaire.
Si la réponse ne contient aucune affirmation factuelle, ne retourne rien.
"""

_CLAIM_VERIFICATION_PROMPT = """Tu vérifies si une affirmation est appuyée par un contexte.

CONTEXTE :
{context}

AFFIRMATION : {claim}

L'affirmation est-elle directement déductible du contexte ci-dessus ?
N'utilise aucune connaissance externe. Ne juge pas si l'affirmation est
vraie dans l'absolu, seulement si le contexte l'appuie.
Réponds par un seul mot : APPUYEE ou NON_APPUYEE
"""

_COVERAGE_PROMPT = """Tu vérifies si une affirmation est couverte par une réponse.

AFFIRMATION : {key_point}
RÉPONSE : {answer}

L'affirmation est-elle exprimée dans la réponse, même reformulée ?
Ignore le style, l'ordre et le vocabulaire. Ne juge que le contenu factuel.
Une affirmation qui porte sur l'ABSENCE d'information n'est couverte que si
la réponse signale explicitement cette absence.
Réponds par un seul mot : COUVERT ou ABSENT
"""


@dataclass
class GenerationCase:
    """Un cas d'evaluation de generation (positif : dans le corpus, ou negatif : hors-corpus).

    Attributes:
        id: Identifiant court du cas.
        question: La question posee au pipeline.
        key_points: Affirmations de reference (verite terrain du gold set,
            IMMUTABLE) que la reponse doit couvrir. Vide pour un cas negatif
            (rien a couvrir hors-corpus).
    """

    id: str
    question: str
    key_points: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    """Sortie du pipeline pour un cas, avant jugement."""

    id: str
    question: str
    answer_text: str
    abstained: bool
    contexts: list[str]
    key_points: list[str]


@dataclass
class FaithfulnessResult:
    """Resultat de `faithfulness` : `score` est None (jamais 1.0) si la
    reponse ne contient aucune affirmation extraite - une reponse vide ne
    contredit jamais le contexte, ce n'est pas la meme chose qu'une reponse
    parfaitement fidele."""

    score: float | None
    n_claims: int


@dataclass
class AnswerCorrectnessResult:
    """Resultat de `answer_correctness` : `uncovered_key_point_ids` est le
    signal le plus actionnable du harness (quelle information precise le
    pipeline perd), a lister par question dans le rapport (etape 5)."""

    score: float | None
    n_covered: int
    n_total: int
    uncovered_key_point_ids: list[str]


def load_gold_cases(path: Path = _DEFAULT_GOLD_PATH) -> list[GenerationCase]:
    """Charge les questions positives du gold set (IMMUTABLE, lecture seule).

    Args:
        path: Chemin du fichier YAML du gold set (meme format que
            `eval/retrieval_eval.py` : id, question, relevant_sources,
            key_points).

    Returns:
        Les cas de generation positifs (un par question du gold set).

    Raises:
        ValueError: Si le fichier est vide ou mal forme.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not raw:
        raise ValueError(f"gold set vide ou introuvable : {path}")

    cases: list[GenerationCase] = []
    for entry in raw:
        try:
            case_id = entry["id"]
            question = entry["question"]
        except KeyError as exc:
            raise ValueError(f"item de gold set invalide (id/question requis) : {entry}") from exc

        key_points = list(entry.get("key_points") or [])
        cases.append(GenerationCase(id=case_id, question=question, key_points=key_points))

    return cases


def negative_cases(questions: list[str] = NEGATIVE_QUESTIONS) -> list[GenerationCase]:
    """Construit les cas negatifs (questions hors-corpus, aucun key_point).

    Args:
        questions: Les questions hors-corpus a tester.

    Returns:
        Les cas de generation negatifs correspondants.
    """
    return [GenerationCase(id=f"neg{i + 1}", question=q, key_points=[]) for i, q in enumerate(questions)]


def run_pipeline_on_cases(pipeline: Any, cases: list[GenerationCase]) -> list[GenerationResult]:
    """Execute le pipeline RAG reel sur chaque cas.

    Args:
        pipeline: Le `Pipeline` (domain/pipeline.py) deja construit.
        cases: Les cas a executer (positifs et/ou negatifs).

    Returns:
        Un `GenerationResult` par cas, dans l'ordre de `cases`.
    """
    results: list[GenerationResult] = []
    for case in cases:
        answer: Answer = pipeline.run(case.question)
        results.append(
            GenerationResult(
                id=case.id,
                question=case.question,
                answer_text=answer.text,
                abstained=answer.abstained,
                contexts=answer.meta.get("selected_chunk_texts", []),
                key_points=case.key_points,
            )
        )
    return results


def load_judge_llm() -> Any:
    """Charge le LLM juge (jamais le generateur RAG local), Anthropic ou Bedrock.

    Le fournisseur est choisi par `LLM_PROVIDER` (.env, charge automatiquement
    ici) : "anthropic" (defaut, retro-compatible) ou "bedrock". Dans les deux
    cas, le resultat est un objet LangChain `BaseChatModel` compatible avec
    `.invoke(prompt)`, configure a `temperature=0` (cf. docstring du module :
    maximum de determinisme disponible, sans garantie bit-a-bit ; `top_p`
    deliberement omis, incompatible avec `temperature` sur certains modeles).

    Returns:
        Une instance `ChatAnthropic` ou `ChatBedrockConverse` configuree.

    Raises:
        RuntimeError: Si `LLM_PROVIDER` est inconnu, ou si une variable
            d'environnement requise par le fournisseur choisi est absente.
    """
    import os

    load_dotenv()

    provider = os.environ.get(_JUDGE_PROVIDER_ENV_VAR, "anthropic").strip().lower()

    if provider == "anthropic":
        return _load_anthropic_judge_llm()
    if provider == "bedrock":
        return _load_bedrock_judge_llm()
    raise RuntimeError(
        f"{_JUDGE_PROVIDER_ENV_VAR} inconnu : {provider!r} "
        "(attendu 'anthropic' ou 'bedrock')"
    )


def judge_model_name() -> str:
    """Nom du modele juge effectivement configure (pour le logguer dans le
    rapport d'ablation - etape 4 : le juge doit rester identifiable et fixe
    pour toute la duree d'une ablation)."""
    import os

    load_dotenv()
    provider = os.environ.get(_JUDGE_PROVIDER_ENV_VAR, "anthropic").strip().lower()
    if provider == "bedrock":
        return os.environ.get(_BEDROCK_MODEL_ID_ENV_VAR, "")
    return os.environ.get(_JUDGE_MODEL_ENV_VAR, "claude-haiku-4-5-20251001")


def _load_anthropic_judge_llm() -> Any:
    """Charge le juge via l'API Anthropic directe (langchain-anthropic).

    Lit `ANTHROPIC_API_KEY` et `ANTHROPIC_JUDGE_MODEL` depuis l'environnement.

    Raises:
        RuntimeError: Si `ANTHROPIC_API_KEY` est absente de l'environnement.
    """
    import os

    from langchain_anthropic import ChatAnthropic

    api_key = os.environ.get(_JUDGE_API_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(
            f"{_JUDGE_API_KEY_ENV_VAR} absente de l'environnement (.env) : "
            "requise pour le juge Claude (LLM_PROVIDER=anthropic)."
        )
    model = os.environ.get(_JUDGE_MODEL_ENV_VAR, "claude-haiku-4-5-20251001")

    return ChatAnthropic(
        model=model, api_key=api_key, temperature=_JUDGE_TEMPERATURE
    )


def _load_bedrock_judge_llm() -> Any:
    """Charge le juge via Amazon Bedrock (langchain-aws, API Converse).

    Lit `AWS_REGION` et `BEDROCK_JUDGE_MODEL_ID` depuis l'environnement et
    verifie la PRESENCE (jamais la valeur) de `AWS_BEARER_TOKEN_BEDROCK` :
    ce jeton n'est jamais lu ni transmis explicitement par ce code - botocore
    (>=1.35) le detecte lui-meme automatiquement dans l'environnement du
    processus pour authentifier les appels Bedrock, ce qui evite de jamais
    faire transiter sa valeur par une variable ou un log applicatif.

    Raises:
        RuntimeError: Si `AWS_BEARER_TOKEN_BEDROCK`, `AWS_REGION` ou
            `BEDROCK_JUDGE_MODEL_ID` sont absentes de l'environnement.
    """
    import os

    from langchain_aws import ChatBedrockConverse

    if not os.environ.get(_BEDROCK_BEARER_TOKEN_ENV_VAR):
        raise RuntimeError(
            f"{_BEDROCK_BEARER_TOKEN_ENV_VAR} absente de l'environnement (.env) : "
            "requise pour le juge Bedrock (LLM_PROVIDER=bedrock)."
        )
    region = os.environ.get(_BEDROCK_REGION_ENV_VAR)
    if not region:
        raise RuntimeError(
            f"{_BEDROCK_REGION_ENV_VAR} absente de l'environnement (.env) : "
            "requise pour le juge Bedrock (LLM_PROVIDER=bedrock)."
        )
    model_id = os.environ.get(_BEDROCK_MODEL_ID_ENV_VAR)
    if not model_id:
        raise RuntimeError(
            f"{_BEDROCK_MODEL_ID_ENV_VAR} absente de l'environnement (.env) : "
            "requise pour le juge Bedrock (LLM_PROVIDER=bedrock)."
        )

    return ChatBedrockConverse(
        model_id=model_id, region_name=region, temperature=_JUDGE_TEMPERATURE
    )


def extract_claims(answer: str, judge: Any) -> list[str]:
    """Decoupe une reponse en affirmations atomiques (un appel LLM).

    Args:
        answer: Le texte de la reponse generee.
        judge: Le LLM juge (cf. `load_judge_llm`).

    Returns:
        Les affirmations extraites (liste vide si aucune affirmation
        factuelle - ex : une abstention pure).
    """
    prompt = _CLAIM_EXTRACTION_PROMPT.format(answer=answer)
    response = judge.invoke(prompt)
    return [line.strip() for line in response.content.strip().splitlines() if line.strip()]


def judge_claim(claim: str, context: str, judge: Any) -> bool:
    """Verifie si une affirmation est appuyee par le contexte (un appel LLM).

    Args:
        claim: L'affirmation a verifier.
        context: Le contexte REELLEMENT fourni au generateur (jamais le
            gold, jamais le corpus entier).
        judge: Le LLM juge.

    Returns:
        True si le juge repond "APPUYEE".
    """
    prompt = _CLAIM_VERIFICATION_PROMPT.format(
        context=context or "(no context retrieved)", claim=claim
    )
    response = judge.invoke(prompt)
    return response.content.strip().upper().startswith("APPUYEE")


def faithfulness(answer: str, context: str, judge: Any) -> FaithfulnessResult:
    """Anti-hallucination : proportion des affirmations de `answer` appuyees par `context`.

    Args:
        answer: Le texte de la reponse generee.
        context: Le contexte REELLEMENT fourni au generateur.
        judge: Le LLM juge.

    Returns:
        `FaithfulnessResult(score=None, n_claims=0)` si aucune affirmation
        n'a ete extraite (jamais `score=1.0` par defaut : une reponse vide
        ne contredit jamais le contexte, ce n'est pas la meme chose qu'une
        reponse parfaitement fidele - cf. docstring de `FaithfulnessResult`).
    """
    claims = extract_claims(answer, judge)
    if not claims:
        return FaithfulnessResult(score=None, n_claims=0)
    supported = sum(1 for claim in claims if judge_claim(claim, context, judge))
    return FaithfulnessResult(score=supported / len(claims), n_claims=len(claims))


def judge_coverage(key_point: str, answer: str, judge: Any) -> bool:
    """Verifie si `key_point` est couvert par `answer`, meme reformule (un appel LLM)."""
    prompt = _COVERAGE_PROMPT.format(key_point=key_point, answer=answer)
    response = judge.invoke(prompt)
    return response.content.strip().upper().startswith("COUVERT")


def answer_correctness(case_id: str, key_points: list[str], answer: str, judge: Any) -> AnswerCorrectnessResult:
    """Anti-omission : proportion des `key_points` du gold couverts par `answer`.

    Args:
        case_id: Identifiant du cas (prefixe des ids de key_point rapportes,
            ex : "q1-kp1").
        key_points: Les affirmations de reference du gold set pour ce cas.
        answer: Le texte de la reponse generee.
        judge: Le LLM juge.

    Returns:
        `AnswerCorrectnessResult(score=None, ...)` si `key_points` est vide
        (cas negatif hors-corpus : rien a couvrir, la metrique ne s'applique
        pas). Sinon le score de couverture et les ids des key_points NON
        couverts (signal le plus actionnable du rapport - etape 5).
    """
    if not key_points:
        return AnswerCorrectnessResult(score=None, n_covered=0, n_total=0, uncovered_key_point_ids=[])

    uncovered: list[str] = []
    covered = 0
    for index, key_point in enumerate(key_points, start=1):
        kp_id = f"{case_id}-kp{index}"
        if judge_coverage(key_point, answer, judge):
            covered += 1
        else:
            uncovered.append(kp_id)

    return AnswerCorrectnessResult(
        score=covered / len(key_points),
        n_covered=covered,
        n_total=len(key_points),
        uncovered_key_point_ids=uncovered,
    )


@dataclass
class CaseJudgment:
    """Jugement complet (les 2 metriques) d'un `GenerationResult`."""

    id: str
    faithfulness: FaithfulnessResult
    correctness: AnswerCorrectnessResult


def evaluate_answer(
    answer: Answer,
    gold_case: GenerationCase,
    judge: Any,
    judge_model: str,
) -> dict[str, Any]:
    """Evaluate a single answer against a gold case.

    Args:
        answer: The Answer object from pipeline.run()
        gold_case: The GenerationCase with question and key_points
        judge: The judge LLM
        judge_model: Judge model name (for logging)

    Returns:
        Dict with faithfulness, answer_correctness, and counts
    """
    # Build context from selected chunks (stored in meta)
    context = "\n\n".join(answer.meta.get("selected_chunk_texts", []))

    # Calculate faithfulness
    faith_result = faithfulness(answer.text, context, judge)

    # Calculate answer correctness
    correctness_result = answer_correctness(
        case_id=gold_case.id,
        key_points=gold_case.key_points,
        answer=answer.text,
        judge=judge,
    )

    return {
        "faithfulness": faith_result.score,
        "n_claims_total": faith_result.n_claims,
        "answer_correctness": correctness_result.score,
        "n_key_points_covered": correctness_result.n_covered,
        "n_key_points_total": correctness_result.n_total,
    }


def judge_results(judge: Any, results: list[GenerationResult]) -> dict[str, CaseJudgment]:
    """Note chaque resultat de generation avec les 2 metriques orthogonales.

    Args:
        judge: Le LLM juge (cf. `load_judge_llm`).
        results: Les resultats de generation a noter.

    Returns:
        {case_id: CaseJudgment}.
    """
    judgments: dict[str, CaseJudgment] = {}
    for result in results:
        context = "\n\n".join(result.contexts)
        judgments[result.id] = CaseJudgment(
            id=result.id,
            faithfulness=faithfulness(result.answer_text, context, judge),
            correctness=answer_correctness(result.id, result.key_points, result.answer_text, judge),
        )
    return judgments


def to_markdown_table(results: list[GenerationResult], judgments: dict[str, CaseJudgment]) -> str:
    """Formate un tableau Markdown recapitulatif (faithfulness + answer_correctness) par cas."""
    lines = [
        "| ID | Abstained | Faithfulness | N Claims | Answer Correctness | KP couverts/total |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        j = judgments[r.id]
        faith = "n/a" if j.faithfulness.score is None else f"{j.faithfulness.score:.2f}"
        correct = "n/a" if j.correctness.score is None else f"{j.correctness.score:.2f}"
        lines.append(
            f"| {r.id} | {r.abstained} | {faith} | {j.faithfulness.n_claims} | "
            f"{correct} | {j.correctness.n_covered}/{j.correctness.n_total} |"
        )
    return "\n".join(lines)


def write_csv(results: list[GenerationResult], judgments: dict[str, CaseJudgment], path: Path) -> None:
    """Ecrit le tableau recapitulatif (faithfulness + answer_correctness) en CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "abstained",
                "faithfulness",
                "n_claims",
                "answer_correctness",
                "n_key_points_covered",
                "n_key_points_total",
                "uncovered_key_point_ids",
            ]
        )
        for r in results:
            j = judgments[r.id]
            writer.writerow(
                [
                    r.id,
                    r.abstained,
                    "" if j.faithfulness.score is None else j.faithfulness.score,
                    j.faithfulness.n_claims,
                    "" if j.correctness.score is None else j.correctness.score,
                    j.correctness.n_covered,
                    j.correctness.n_total,
                    ";".join(j.correctness.uncovered_key_point_ids),
                ]
            )


def main() -> None:
    """Charge la config + le gold set reels, evalue la generation, ecrit les rapports."""
    from src.application.answer import build_pipeline
    from src.cli import load_chunks, load_config, load_system_prompt

    config = load_config()

    chunks_path = Path(config["vectorstore"]["chunks_path"])
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"index introuvable ({chunks_path}) : executer "
            "`uv run python -m src.cli index` avant l'evaluation"
        )
    chunks = load_chunks(chunks_path)
    system_prompt = load_system_prompt()

    pipeline = build_pipeline(config, chunks, system_prompt)

    cases = load_gold_cases() + negative_cases()
    results = run_pipeline_on_cases(pipeline, cases)

    judge = load_judge_llm()
    judgments = judge_results(judge, results)

    print(to_markdown_table(results, judgments))

    write_csv(results, judgments, _DEFAULT_RESULTS_CSV_PATH)
    print(f"\nResultats CSV ecrits dans {_DEFAULT_RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
