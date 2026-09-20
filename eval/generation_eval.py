"""Evalue la generation du pipeline RAG : juge LLM-as-judge custom (faithfulness,
relevancy) + integration RAGAS (faithfulness, answer_relevancy, context_precision,
context_recall).

Le systeme RAG reste 100% LOCAL (Ollama) : SEUL le juge utilise une API Claude
distante (jamais le petit modele local 1.7-2B teste comme generateur).

Le fournisseur du juge est choisi par la variable d'environnement
LLM_PROVIDER (.env, jamais committee) :
- "anthropic" (defaut, retro-compatible) : API Anthropic directe via
  langchain-anthropic (`ChatAnthropic`), modele configurable via
  ANTHROPIC_JUDGE_MODEL, cle lue depuis ANTHROPIC_API_KEY.
- "bedrock" : Amazon Bedrock (API Converse) via langchain-aws
  (`ChatBedrockConverse`), modele lu depuis BEDROCK_JUDGE_MODEL_ID, region
  depuis AWS_REGION. Authentification par jeton porteur
  (AWS_BEARER_TOKEN_BEDROCK) : ce code ne lit QUE sa presence, jamais sa
  valeur - botocore le detecte lui-meme dans l'environnement du processus
  (cf. `_load_bedrock_judge_llm`), qui n'est donc jamais logue ni transmis
  explicitement en Python.
Voir `load_judge_llm` pour le detail de la selection.

Note empirique (verifiee en direct) : claude-sonnet-5 et claude-opus-5
rejettent le parametre `temperature` que RAGAS envoie en interne pour
plusieurs metriques (erreur API "temperature is deprecated for this model") ;
claude-haiku-4-5-20251001 l'accepte normalement. C'est pourquoi
ANTHROPIC_JUDGE_MODEL vaut claude-haiku-4-5-20251001 par defaut.

RAGAS appelle OpenAI par defaut : on lui passe EXPLICITEMENT le LLM Claude
(via `ragas.llms.LangchainLLMWrapper`) et un embedder LOCAL sentence-
transformers (le meme BgeM3Embedder que le systeme RAG, cf. embedding.bge_m3
en config), sinon il tenterait un appel OpenAI. Note technique (verifiee en
direct) : la metrique `answer_relevancy` de RAGAS attend l'interface
LangChain classique (`embed_query`/`embed_documents`), pas l'interface
native `ragas.embeddings` (`embed_text`/`embed_texts`) : on enveloppe donc
notre Embedder via un petit adaptateur avant de le passer a
`ragas.embeddings.LangchainEmbeddingsWrapper`.

Usage:
    uv run python -m eval.generation_eval
"""

import csv
import json
import re
from dataclasses import dataclass
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

# Questions hors-corpus : le pipeline doit s'abstenir sur chacune d'elles.
NEGATIVE_QUESTIONS = [
    "What is the capital of France?",
    "Who won the FIFA World Cup in 2022?",
]

_JUDGE_PROMPT_TEMPLATE = """You are a strict evaluation judge for a RAG (Retrieval-Augmented Generation) system.

Given a QUESTION, the CONTEXT that was retrieved to answer it, and the ANSWER the system produced, score two things:

1. "faithfulness" (0.0 to 1.0): does every factual claim in ANSWER come from CONTEXT, without hallucination or unsupported invention? An honest abstention ("Information non trouvee dans les documents.") is always fully faithful (1.0), since it invents nothing.
2. "relevancy" (0.0 to 1.0): does ANSWER actually address QUESTION? An abstention is relevant (high score) ONLY if CONTEXT genuinely does not contain the answer; it is NOT relevant (low score) if CONTEXT did contain a usable answer but the system abstained anyway.

Respond with ONLY a JSON object, no other text, no markdown code fences:
{{"faithfulness": <float 0.0-1.0>, "relevancy": <float 0.0-1.0>, "reasoning": "<one short sentence>"}}

QUESTION: {question}

CONTEXT:
{context}

ANSWER: {answer}
"""


@dataclass
class GenerationCase:
    """Un cas d'evaluation de generation (positif : dans le corpus, ou negatif : hors-corpus).

    Attributes:
        id: Identifiant court du cas.
        question: La question posee au pipeline.
        reference: Reponse de reference (ex: key_points du gold set joints),
            utilisee par RAGAS pour context_precision/context_recall. None
            pour un cas negatif (aucune reponse de reference n'existe).
    """

    id: str
    question: str
    reference: str | None = None


@dataclass
class GenerationResult:
    """Sortie du pipeline pour un cas, avant jugement."""

    id: str
    question: str
    answer_text: str
    abstained: bool
    contexts: list[str]
    reference: str | None


@dataclass
class JudgeScore:
    """Score du juge LLM custom pour une reponse."""

    faithfulness: float
    relevancy: float
    reasoning: str


def load_gold_cases(path: Path = _DEFAULT_GOLD_PATH) -> list[GenerationCase]:
    """Charge les questions positives du gold set (reference = key_points joints).

    Args:
        path: Chemin du fichier YAML du gold set (meme format que
            `eval/retrieval_eval.py` : id, question, relevant_sources,
            key_points optionnels).

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

        key_points = entry.get("key_points") or []
        reference = " ".join(key_points) if key_points else None
        cases.append(GenerationCase(id=case_id, question=question, reference=reference))

    return cases


def negative_cases(questions: list[str] = NEGATIVE_QUESTIONS) -> list[GenerationCase]:
    """Construit les cas negatifs (questions hors-corpus, aucune reference).

    Args:
        questions: Les questions hors-corpus a tester.

    Returns:
        Les cas de generation negatifs correspondants.
    """
    return [GenerationCase(id=f"neg{i + 1}", question=q) for i, q in enumerate(questions)]


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
                reference=case.reference,
            )
        )
    return results


def load_judge_llm() -> Any:
    """Charge le LLM juge (jamais le generateur RAG local), Anthropic ou Bedrock.

    Le fournisseur est choisi par `LLM_PROVIDER` (.env, charge automatiquement
    ici) : "anthropic" (defaut, retro-compatible) ou "bedrock". Dans les deux
    cas, le resultat est un objet LangChain `BaseChatModel` compatible avec
    `judge_answer()` (appel direct `.invoke(prompt)`) et avec
    `ragas.llms.LangchainLLMWrapper` (cf. `load_ragas_judge_and_embeddings`).

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

    return ChatAnthropic(model=model, api_key=api_key)


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

    return ChatBedrockConverse(model_id=model_id, region_name=region)


def _parse_judge_json(raw_text: str) -> dict[str, Any]:
    """Extrait le JSON de la reponse du juge (tolere d'eventuelles balises markdown).

    Args:
        raw_text: Le texte brut renvoye par le juge.

    Returns:
        Le dict JSON parse.

    Raises:
        ValueError: Si aucun JSON valide n'a pu etre extrait.
    """
    text = raw_text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"reponse du juge sans JSON exploitable : {raw_text!r}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON du juge invalide : {raw_text!r}") from exc


def judge_answer(judge_llm: Any, question: str, context: str, answer: str) -> JudgeScore:
    """Note une reponse (faithfulness, relevancy) via le juge LLM custom.

    Args:
        judge_llm: Le LLM juge (cf. `load_judge_llm`).
        question: La question posee.
        context: Le contexte fourni au generateur (chunks concatenes).
        answer: La reponse produite par le generateur.

    Returns:
        Le `JudgeScore` (faithfulness, relevancy, reasoning).
    """
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=question, context=context or "(no context retrieved)", answer=answer
    )
    response = judge_llm.invoke(prompt)
    payload = _parse_judge_json(response.content)

    return JudgeScore(
        faithfulness=float(payload["faithfulness"]),
        relevancy=float(payload["relevancy"]),
        reasoning=str(payload.get("reasoning", "")),
    )


def custom_judge_eval(judge_llm: Any, results: list[GenerationResult]) -> dict[str, JudgeScore]:
    """Note chaque resultat de generation avec le juge LLM custom.

    Args:
        judge_llm: Le LLM juge (cf. `load_judge_llm`).
        results: Les resultats de generation a noter.

    Returns:
        {case_id: JudgeScore}.
    """
    return {
        r.id: judge_answer(judge_llm, r.question, "\n\n".join(r.contexts), r.answer_text)
        for r in results
    }


class _LangchainCompatibleEmbeddings:
    """Adapte un `Embedder` du projet a l'interface LangChain (embed_query/embed_documents).

    RAGAS exige cette interface pour `answer_relevancy` (verifie en direct) ;
    elle n'existe pas sur le port `Embedder` du projet (`embed`/`embed_query`).
    """

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts)


def load_ragas_judge_and_embeddings(embedding_config: dict[str, Any]) -> tuple[Any, Any]:
    """Construit le LLM juge et l'embedder LOCAL passes explicitement a RAGAS.

    Args:
        embedding_config: Section de configuration `embedding` (cf.
            `config/default.yaml`), utilisee pour charger le MEME embedder
            local que le systeme RAG (jamais OpenAI).

    Returns:
        (llm, embeddings) prets a passer a `ragas.evaluate(..., llm=, embeddings=)`.
    """
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from src.adapters.embedding.factory import create_embedder

    judge_llm = LangchainLLMWrapper(load_judge_llm())
    embedder = create_embedder(embedding_config)
    judge_embeddings = LangchainEmbeddingsWrapper(_LangchainCompatibleEmbeddings(embedder))

    return judge_llm, judge_embeddings


def ragas_eval(
    results: list[GenerationResult], judge_llm: Any, judge_embeddings: Any
) -> dict[str, dict[str, float]]:
    """Evalue les resultats de generation via RAGAS (faithfulness, answer_relevancy,
    context_precision, context_recall).

    Les deux dernieres metriques necessitent une `reference` : les cas qui
    n'en ont pas (ex: cas negatifs hors-corpus) ne recoivent que
    faithfulness/answer_relevancy.

    Args:
        results: Les resultats de generation a evaluer.
        judge_llm: LLM juge deja enveloppe pour RAGAS (cf.
            `load_ragas_judge_and_embeddings`).
        judge_embeddings: Embedder local deja enveloppe pour RAGAS.

    Returns:
        {case_id: {metric_name: score}}.
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    scores: dict[str, dict[str, float]] = {r.id: {} for r in results}

    with_reference = [r for r in results if r.reference]
    without_reference = [r for r in results if not r.reference]

    if with_reference:
        dataset = EvaluationDataset.from_list(
            [
                {
                    "user_input": r.question,
                    "response": r.answer_text,
                    "retrieved_contexts": r.contexts or [""],
                    "reference": r.reference,
                }
                for r in with_reference
            ]
        )
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=judge_llm,
            embeddings=judge_embeddings,
        )
        for r, row in zip(with_reference, result.to_pandas().to_dict(orient="records")):
            scores[r.id].update(
                {
                    "faithfulness": row["faithfulness"],
                    "answer_relevancy": row["answer_relevancy"],
                    "context_precision": row["context_precision"],
                    "context_recall": row["context_recall"],
                }
            )

    if without_reference:
        dataset = EvaluationDataset.from_list(
            [
                {
                    "user_input": r.question,
                    "response": r.answer_text,
                    "retrieved_contexts": r.contexts or [""],
                }
                for r in without_reference
            ]
        )
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=judge_llm,
            embeddings=judge_embeddings,
        )
        for r, row in zip(without_reference, result.to_pandas().to_dict(orient="records")):
            scores[r.id].update(
                {"faithfulness": row["faithfulness"], "answer_relevancy": row["answer_relevancy"]}
            )

    return scores


def to_markdown_table(
    results: list[GenerationResult],
    custom_scores: dict[str, JudgeScore],
    ragas_scores: dict[str, dict[str, float]],
) -> str:
    """Formate un tableau Markdown recapitulatif (juge custom + RAGAS) par cas."""
    lines = [
        "| ID | Abstained | Faithfulness (custom) | Relevancy (custom) | "
        "Faithfulness (RAGAS) | Answer Relevancy | Context Precision | Context Recall |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        c = custom_scores.get(r.id)
        g = ragas_scores.get(r.id, {})
        lines.append(
            f"| {r.id} | {r.abstained} | "
            f"{c.faithfulness:.2f} | {c.relevancy:.2f} | "
            f"{g.get('faithfulness', float('nan')):.2f} | "
            f"{g.get('answer_relevancy', float('nan')):.2f} | "
            f"{g.get('context_precision', float('nan')):.2f} | "
            f"{g.get('context_recall', float('nan')):.2f} |"
        )
    return "\n".join(lines)


def write_csv(
    results: list[GenerationResult],
    custom_scores: dict[str, JudgeScore],
    ragas_scores: dict[str, dict[str, float]],
    path: Path,
) -> None:
    """Ecrit le tableau recapitulatif (juge custom + RAGAS) en CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "abstained",
                "faithfulness_custom",
                "relevancy_custom",
                "faithfulness_ragas",
                "answer_relevancy_ragas",
                "context_precision_ragas",
                "context_recall_ragas",
            ]
        )
        for r in results:
            c = custom_scores.get(r.id)
            g = ragas_scores.get(r.id, {})
            writer.writerow(
                [
                    r.id,
                    r.abstained,
                    c.faithfulness,
                    c.relevancy,
                    g.get("faithfulness", ""),
                    g.get("answer_relevancy", ""),
                    g.get("context_precision", ""),
                    g.get("context_recall", ""),
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

    judge_llm_raw = load_judge_llm()
    custom_scores = custom_judge_eval(judge_llm_raw, results)

    ragas_llm, ragas_embeddings = load_ragas_judge_and_embeddings(config["embedding"])
    ragas_scores = ragas_eval(results, ragas_llm, ragas_embeddings)

    print(to_markdown_table(results, custom_scores, ragas_scores))

    write_csv(results, custom_scores, ragas_scores, _DEFAULT_RESULTS_CSV_PATH)
    print(f"\nResultats CSV ecrits dans {_DEFAULT_RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
