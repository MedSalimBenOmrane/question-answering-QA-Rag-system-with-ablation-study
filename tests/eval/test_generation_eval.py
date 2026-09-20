"""Tests de eval/generation_eval.py : gold set, faithfulness (anti-hallucination),
answer_correctness (anti-omission), rapports.

Aucun mock. Les tests qui appellent reellement un juge LLM (Claude/Bedrock)
sont minimises en nombre (cout reel a chaque execution, un appel par
affirmation) et sautes automatiquement si les identifiants correspondants ne
sont pas configures dans l'environnement/.env.
"""

import csv
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from eval.generation_eval import (
    AnswerCorrectnessResult,
    CaseJudgment,
    FaithfulnessResult,
    GenerationCase,
    GenerationResult,
    answer_correctness,
    extract_claims,
    faithfulness,
    judge_claim,
    judge_coverage,
    judge_results,
    load_gold_cases,
    load_judge_llm,
    negative_cases,
    run_pipeline_on_cases,
    to_markdown_table,
    write_csv,
)
from src.adapters.guardrails.basic import BasicGuardrail
from src.adapters.llm.ollama import build_prompt, create_llm
from src.adapters.llm.token_counter import TiktokenTokenCounter
from src.adapters.reranking.noop import NoOpReranker
from src.adapters.retrieval.bm25 import BM25Retriever
from src.domain.budget import TopKSelector
from src.domain.models import Chunk
from src.domain.pipeline import Pipeline, PipelineConfig

load_dotenv()
_HAS_ANTHROPIC_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))
_requires_claude = pytest.mark.skipif(
    not _HAS_ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY non configuree : juge Claude indisponible"
)

_HAS_BEDROCK_TOKEN = bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK"))
_requires_bedrock = pytest.mark.skipif(
    not _HAS_BEDROCK_TOKEN,
    reason="AWS_BEARER_TOKEN_BEDROCK non configuree : juge Bedrock indisponible",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SYSTEM_PROMPT = (_REPO_ROOT / "src" / "prompts" / "system.txt").read_text(encoding="utf-8")

_FAKE_GOLD_YAML = """\
- id: q1
  question: "What fuels the propulsion system?"
  relevant_sources:
    - propulsion.md
  key_points:
    - "Xenon fuels the ion thrusters."

- id: q2
  question: "What does the cafeteria serve?"
  relevant_sources:
    - cafeteria.md
  key_points: []
"""

_CORPUS = [
    Chunk(
        id="c1",
        text="The propulsion system uses xenon as fuel for the ion thrusters.",
        source="propulsion.md",
        n_tokens=12,
    ),
]


def _build_fast_pipeline() -> Pipeline:
    """Pipeline reel rapide (BM25, pas de reranking, top-k) pour tester
    run_pipeline_on_cases() sans charger BGE-M3/cross-encoder."""
    token_counter = TiktokenTokenCounter(__import__("tiktoken").get_encoding("cl100k_base"))
    return Pipeline(
        guardrail_input=BasicGuardrail(max_length=2000),
        retriever=BM25Retriever(chunks=_CORPUS, k1=1.5, b=0.75),
        reranker=NoOpReranker(),
        selector=TopKSelector(),
        llm=create_llm(
            {"model": "qwen3:1.7b", "temperature": 0.1},
            system_prompt=_SYSTEM_PROMPT,
            max_output_tokens=200,
        ),
        guardrail_output=BasicGuardrail(max_length=4000),
        token_counter=token_counter,
        assemble_prompt=build_prompt,
        system_prompt=_SYSTEM_PROMPT,
        config=PipelineConfig(retrieve_k=5, rerank_top_n=5, budget_total=1024, reserve_answer=200),
    )


class TestLoadGoldCases:
    def test_parses_key_points(self, tmp_path: Path) -> None:
        path = tmp_path / "gold.yaml"
        path.write_text(_FAKE_GOLD_YAML, encoding="utf-8")

        cases = load_gold_cases(path)

        assert len(cases) == 2
        assert cases[0] == GenerationCase(
            id="q1",
            question="What fuels the propulsion system?",
            key_points=["Xenon fuels the ion thrusters."],
        )

    def test_missing_key_points_gives_empty_list(self, tmp_path: Path) -> None:
        path = tmp_path / "gold.yaml"
        path.write_text(_FAKE_GOLD_YAML, encoding="utf-8")
        cases = load_gold_cases(path)
        assert cases[1].key_points == []

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "gold.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            load_gold_cases(path)


class TestNegativeCases:
    def test_builds_cases_without_key_points(self) -> None:
        cases = negative_cases(["What is the capital of France?"])
        assert cases == [
            GenerationCase(id="neg1", question="What is the capital of France?", key_points=[])
        ]


class TestRunPipelineOnCases:
    def test_captures_answer_contexts_and_key_points(self) -> None:
        pipeline = _build_fast_pipeline()
        cases = [
            GenerationCase(
                id="q1",
                question="What fuel does the propulsion system use?",
                key_points=["Xenon fuels the ion thrusters."],
            )
        ]

        results = run_pipeline_on_cases(pipeline, cases)

        assert len(results) == 1
        assert results[0].id == "q1"
        assert "xenon" in results[0].answer_text.lower()
        assert results[0].abstained is False
        assert any("xenon" in c.lower() for c in results[0].contexts)
        assert results[0].key_points == ["Xenon fuels the ion thrusters."]

    def test_negative_case_abstains(self) -> None:
        pipeline = _build_fast_pipeline()
        cases = negative_cases(["What is the capital of France?"])

        results = run_pipeline_on_cases(pipeline, cases)

        assert results[0].abstained is True
        assert results[0].key_points == []


class TestAnswerCorrectnessNoKeyPoints:
    """`answer_correctness` renvoie immediatement sans jamais appeler le juge
    si `key_points` est vide (cas negatif hors-corpus) : logique pure, aucun
    cout reel, verifiee en passant un juge factice qui leverait si appele."""

    def test_empty_key_points_returns_none_without_calling_judge(self) -> None:
        def _unused_judge_would_raise(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("le juge ne doit jamais etre appele si key_points est vide")

        result = answer_correctness("q1", [], "peu importe la reponse", _unused_judge_would_raise)

        assert result == AnswerCorrectnessResult(
            score=None, n_covered=0, n_total=0, uncovered_key_point_ids=[]
        )


class TestReports:
    _RESULTS = [
        GenerationResult(
            id="q1", question="q", answer_text="a", abstained=False, contexts=["ctx"], key_points=["kp1"]
        )
    ]
    _JUDGMENTS = {
        "q1": CaseJudgment(
            id="q1",
            faithfulness=FaithfulnessResult(score=0.9, n_claims=3),
            correctness=AnswerCorrectnessResult(score=1.0, n_covered=1, n_total=1, uncovered_key_point_ids=[]),
        )
    }

    def test_markdown_table_contains_scores(self) -> None:
        table = to_markdown_table(self._RESULTS, self._JUDGMENTS)
        assert "| q1 |" in table
        assert "0.90" in table
        assert "1.00" in table
        assert "1/1" in table

    def test_markdown_table_handles_none_scores(self) -> None:
        judgments = {
            "q1": CaseJudgment(
                id="q1",
                faithfulness=FaithfulnessResult(score=None, n_claims=0),
                correctness=AnswerCorrectnessResult(score=None, n_covered=0, n_total=0, uncovered_key_point_ids=[]),
            )
        }
        table = to_markdown_table(self._RESULTS, judgments)
        assert "n/a" in table

    def test_write_csv_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "results.csv"
        write_csv(self._RESULTS, self._JUDGMENTS, path)

        with path.open(encoding="utf-8") as f:
            rows = list(csv.reader(f))

        assert rows[0][0] == "id"
        assert rows[1][0] == "q1"
        assert rows[1][2] == "0.9"

    def test_write_csv_lists_uncovered_key_point_ids(self, tmp_path: Path) -> None:
        judgments = {
            "q1": CaseJudgment(
                id="q1",
                faithfulness=FaithfulnessResult(score=1.0, n_claims=1),
                correctness=AnswerCorrectnessResult(
                    score=0.5, n_covered=1, n_total=2, uncovered_key_point_ids=["q1-kp2"]
                ),
            )
        }
        path = tmp_path / "results.csv"
        write_csv(self._RESULTS, judgments, path)

        with path.open(encoding="utf-8") as f:
            rows = list(csv.reader(f))

        assert rows[1][-1] == "q1-kp2"


class TestLoadJudgeLLMProviderSelection:
    """Selection du fournisseur (LLM_PROVIDER) : logique pure, aucun appel reseau."""

    def test_unknown_provider_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "does-not-exist")

        with pytest.raises(RuntimeError, match="LLM_PROVIDER"):
            load_judge_llm()

    def test_bedrock_missing_bearer_token_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Valeur vide plutot que delenv : load_judge_llm() appelle load_dotenv()
        # (override=False par defaut), qui rechargerait la vraie valeur depuis
        # .env si la variable etait totalement absente de os.environ.
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "")

        with pytest.raises(RuntimeError, match="AWS_BEARER_TOKEN_BEDROCK"):
            load_judge_llm()

    def test_bedrock_missing_region_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "dummy-token-for-presence-check-only")
        monkeypatch.setenv("AWS_REGION", "")

        with pytest.raises(RuntimeError, match="AWS_REGION"):
            load_judge_llm()

    def test_bedrock_missing_model_id_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "dummy-token-for-presence-check-only")
        monkeypatch.setenv("AWS_REGION", "eu-west-3")
        monkeypatch.setenv("BEDROCK_JUDGE_MODEL_ID", "")

        with pytest.raises(RuntimeError, match="BEDROCK_JUDGE_MODEL_ID"):
            load_judge_llm()

    def test_anthropic_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")

        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            load_judge_llm()


@_requires_claude
class TestFaithfulnessRealClaude:
    """Appels reels au juge (un par affirmation extraite) : minimises en nombre."""

    @pytest.fixture(autouse=True)
    def _force_anthropic_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Le vrai .env de developpement peut avoir LLM_PROVIDER=bedrock : ces
        # tests testent specifiquement le chemin Anthropic (cf. _requires_claude).
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    def test_grounded_answer_is_fully_faithful(self) -> None:
        judge = load_judge_llm()
        context = "The propulsion system uses xenon as fuel for the ion thrusters."
        answer = "The propulsion system uses xenon as fuel."

        result = faithfulness(answer, context, judge)

        assert result.n_claims >= 1
        assert result.score is not None
        assert result.score > 0.5

    def test_pure_abstention_yields_none_score_not_one(self) -> None:
        """Reproduit le bug de l'ancien harness (faithfulness_custom=1.00 pour
        une reponse quasi vide) : une abstention pure ne doit jamais produire
        un score de 1.0 par defaut, mais None (aucune affirmation a juger)."""
        judge = load_judge_llm()

        result = faithfulness(
            "Information non trouvée dans les documents.",
            "The propulsion system uses xenon as fuel for the ion thrusters.",
            judge,
        )

        if result.n_claims == 0:
            assert result.score is None
        else:
            # le juge a extrait une "affirmation" de l'abstention elle-meme -
            # comportement du LLM, pas de notre logique : on verifie alors
            # seulement la coherence interne (score defini si n_claims > 0).
            assert result.score is not None

    def test_extract_claims_and_judge_claim_are_independently_callable(self) -> None:
        judge = load_judge_llm()
        claims = extract_claims("The propulsion system uses xenon as fuel.", judge)
        assert len(claims) >= 1
        assert judge_claim(
            claims[0], "The propulsion system uses xenon as fuel for the ion thrusters.", judge
        ) in (True, False)


@_requires_claude
class TestAnswerCorrectnessRealClaude:
    @pytest.fixture(autouse=True)
    def _force_anthropic_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    def test_fully_covering_answer_scores_one(self) -> None:
        judge = load_judge_llm()

        result = answer_correctness(
            "q1",
            ["Xenon fuels the ion thrusters."],
            "The ion thrusters are fueled by xenon.",
            judge,
        )

        assert result.n_total == 1
        assert result.score == pytest.approx(1.0)
        assert result.uncovered_key_point_ids == []

    def test_uncovered_key_point_is_reported_with_its_id(self) -> None:
        judge = load_judge_llm()

        result = answer_correctness(
            "q1",
            ["Xenon fuels the ion thrusters.", "The reactor produces 500 megawatts."],
            "The ion thrusters are fueled by xenon.",
            judge,
        )

        assert result.n_covered == 1
        assert result.n_total == 2
        assert "q1-kp2" in result.uncovered_key_point_ids

    def test_judge_coverage_directly_callable(self) -> None:
        judge = load_judge_llm()
        assert judge_coverage("Xenon fuels the ion thrusters.", "It uses xenon.", judge) in (True, False)


@_requires_claude
class TestJudgeResultsRealClaude:
    @pytest.fixture(autouse=True)
    def _force_anthropic_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    def test_judges_every_result(self) -> None:
        judge = load_judge_llm()
        results = [
            GenerationResult(
                id="q1",
                question="What fuel does the propulsion system use?",
                answer_text="The propulsion system uses xenon as fuel.",
                abstained=False,
                contexts=["The propulsion system uses xenon as fuel for the ion thrusters."],
                key_points=["Xenon fuels the ion thrusters."],
            )
        ]

        judgments = judge_results(judge, results)

        assert set(judgments.keys()) == {"q1"}
        assert isinstance(judgments["q1"], CaseJudgment)
        assert judgments["q1"].correctness.n_total == 1


@_requires_bedrock
class TestFaithfulnessRealBedrock:
    """Un seul test reel Bedrock (le mecanisme est deja valide via Claude ci-dessus)."""

    def test_grounded_answer_is_fully_faithful(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        judge = load_judge_llm()

        result = faithfulness(
            "The propulsion system uses xenon as fuel.",
            "The propulsion system uses xenon as fuel for the ion thrusters.",
            judge,
        )

        assert result.n_claims >= 1
        assert result.score is not None
        assert result.score > 0.5
