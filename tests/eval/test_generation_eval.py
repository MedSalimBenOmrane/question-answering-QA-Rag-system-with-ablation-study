"""Tests de eval/generation_eval.py : gold set, juge custom, RAGAS, rapports.

Aucun mock. Les tests qui appellent reellement l'API Claude sont minimises en
nombre (cout reel a chaque execution) et sautes automatiquement si
ANTHROPIC_API_KEY n'est pas configuree dans l'environnement/.env.
"""

import csv
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from eval.generation_eval import (
    GenerationCase,
    GenerationResult,
    JudgeScore,
    _LangchainCompatibleEmbeddings,
    _parse_judge_json,
    custom_judge_eval,
    judge_answer,
    load_gold_cases,
    load_judge_llm,
    negative_cases,
    ragas_eval,
    run_pipeline_on_cases,
    to_markdown_table,
    write_csv,
)
from src.adapters.embedding.bge_m3 import BgeM3Embedder
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
        llm=create_llm({"model": "qwen3:1.7b", "temperature": 0.1}, system_prompt=_SYSTEM_PROMPT),
        guardrail_output=BasicGuardrail(max_length=4000),
        token_counter=token_counter,
        assemble_prompt=build_prompt,
        system_prompt=_SYSTEM_PROMPT,
        config=PipelineConfig(retrieve_k=5, rerank_top_n=5, budget_total=1024, reserve_answer=200),
    )


class TestLoadGoldCases:
    def test_parses_key_points_into_reference(self, tmp_path: Path) -> None:
        path = tmp_path / "gold.yaml"
        path.write_text(_FAKE_GOLD_YAML, encoding="utf-8")

        cases = load_gold_cases(path)

        assert len(cases) == 2
        assert cases[0] == GenerationCase(
            id="q1", question="What fuels the propulsion system?", reference="Xenon fuels the ion thrusters."
        )

    def test_missing_key_points_gives_none_reference(self, tmp_path: Path) -> None:
        path = tmp_path / "gold.yaml"
        path.write_text(_FAKE_GOLD_YAML, encoding="utf-8")
        cases = load_gold_cases(path)
        assert cases[1].reference is None

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "gold.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            load_gold_cases(path)


class TestNegativeCases:
    def test_builds_cases_without_reference(self) -> None:
        cases = negative_cases(["What is the capital of France?"])
        assert cases == [GenerationCase(id="neg1", question="What is the capital of France?", reference=None)]


class TestParseJudgeJson:
    def test_parses_clean_json(self) -> None:
        payload = _parse_judge_json('{"faithfulness": 0.9, "relevancy": 0.8, "reasoning": "ok"}')
        assert payload == {"faithfulness": 0.9, "relevancy": 0.8, "reasoning": "ok"}

    def test_parses_json_wrapped_in_markdown_fence(self) -> None:
        raw = '```json\n{"faithfulness": 1.0, "relevancy": 1.0, "reasoning": "fine"}\n```'
        payload = _parse_judge_json(raw)
        assert payload["faithfulness"] == 1.0

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_judge_json("not json at all")


class TestRunPipelineOnCases:
    def test_captures_answer_and_contexts(self) -> None:
        pipeline = _build_fast_pipeline()
        cases = [GenerationCase(id="q1", question="What fuel does the propulsion system use?")]

        results = run_pipeline_on_cases(pipeline, cases)

        assert len(results) == 1
        assert results[0].id == "q1"
        assert "xenon" in results[0].answer_text.lower()
        assert results[0].abstained is False
        assert any("xenon" in c.lower() for c in results[0].contexts)

    def test_negative_case_abstains(self) -> None:
        pipeline = _build_fast_pipeline()
        cases = negative_cases(["What is the capital of France?"])

        results = run_pipeline_on_cases(pipeline, cases)

        assert results[0].abstained is True


class TestLangchainCompatibleEmbeddings:
    def test_delegates_to_real_embedder(self, bge_m3_embedder: BgeM3Embedder) -> None:
        shim = _LangchainCompatibleEmbeddings(bge_m3_embedder)

        query_vector = shim.embed_query("hello world")
        doc_vectors = shim.embed_documents(["hello world", "another doc"])

        assert len(query_vector) == 1024
        assert len(doc_vectors) == 2
        assert len(doc_vectors[0]) == 1024


class TestReports:
    _RESULTS = [
        GenerationResult(
            id="q1", question="q", answer_text="a", abstained=False, contexts=["ctx"], reference="ref"
        )
    ]
    _CUSTOM = {"q1": JudgeScore(faithfulness=0.9, relevancy=0.8, reasoning="ok")}
    _RAGAS = {
        "q1": {
            "faithfulness": 0.95,
            "answer_relevancy": 0.85,
            "context_precision": 1.0,
            "context_recall": 1.0,
        }
    }

    def test_markdown_table_contains_scores(self) -> None:
        table = to_markdown_table(self._RESULTS, self._CUSTOM, self._RAGAS)
        assert "| q1 |" in table
        assert "0.90" in table
        assert "0.95" in table

    def test_write_csv_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "results.csv"
        write_csv(self._RESULTS, self._CUSTOM, self._RAGAS, path)

        with path.open(encoding="utf-8") as f:
            rows = list(csv.reader(f))

        assert rows[0][0] == "id"
        assert rows[1][0] == "q1"
        assert rows[1][2] == "0.9"


@_requires_claude
class TestJudgeAnswerRealClaude:
    """Appels reels a l'API Claude (juge), minimises en nombre."""

    def test_judge_scores_a_faithful_grounded_answer_highly(self) -> None:
        judge_llm = load_judge_llm()

        score = judge_answer(
            judge_llm,
            question="What fuel does the propulsion system use?",
            context="The propulsion system uses xenon as fuel for the ion thrusters.",
            answer="The propulsion system uses xenon as fuel.",
        )

        assert isinstance(score, JudgeScore)
        assert 0.0 <= score.faithfulness <= 1.0
        assert 0.0 <= score.relevancy <= 1.0
        assert score.faithfulness > 0.5

    def test_judge_scores_honest_abstention_as_faithful(self) -> None:
        judge_llm = load_judge_llm()

        score = judge_answer(
            judge_llm,
            question="What is the capital of France?",
            context="The propulsion system uses xenon as fuel for the ion thrusters.",
            answer="Information non trouvée dans les documents.",
        )

        assert score.faithfulness > 0.5

    def test_custom_judge_eval_scores_each_result(self) -> None:
        judge_llm = load_judge_llm()
        results = [
            GenerationResult(
                id="q1",
                question="What fuel does the propulsion system use?",
                answer_text="The propulsion system uses xenon as fuel.",
                abstained=False,
                contexts=["The propulsion system uses xenon as fuel for the ion thrusters."],
                reference="Xenon fuels the ion thrusters.",
            )
        ]

        scores = custom_judge_eval(judge_llm, results)

        assert set(scores.keys()) == {"q1"}
        assert isinstance(scores["q1"], JudgeScore)


@_requires_claude
class TestRagasEvalReal:
    """Un seul appel reel bout en bout (4 metriques) : le juge + l'embedder
    local sont deja valides individuellement par les tests ci-dessus."""

    def test_ragas_eval_returns_all_four_metrics_with_reference(self) -> None:
        from eval.generation_eval import load_ragas_judge_and_embeddings

        results = [
            GenerationResult(
                id="q1",
                question="What fuel does the propulsion system use?",
                answer_text="The propulsion system uses xenon as fuel for the ion thrusters.",
                abstained=False,
                contexts=["The propulsion system uses xenon as fuel for the ion thrusters."],
                reference="Xenon fuels the ion thrusters.",
            )
        ]

        judge_llm, judge_embeddings = load_ragas_judge_and_embeddings(
            {"provider": "bge_m3", "bge_m3": {"model_name": "BAAI/bge-m3", "device": "cpu", "batch_size": 32}}
        )

        scores = ragas_eval(results, judge_llm, judge_embeddings)

        assert set(scores["q1"].keys()) == {
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        }
        for value in scores["q1"].values():
            assert 0.0 <= value <= 1.0
