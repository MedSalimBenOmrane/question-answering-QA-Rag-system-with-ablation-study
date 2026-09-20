"""Tests de eval/retrieval_eval.py : to_sources, les 4 metriques, gold set,
evaluation a partir d'une Answer reelle, rapports.

Utilise un gold set FACTICE (pas eval/gold_retrieval.yaml, IMMUTABLE et
reserve au vrai corpus de l'utilisateur) et un pipeline reel minimal (BM25 +
NoOpReranker + TopKSelector + Ollama) sur un mini-corpus en memoire pour le
seul test de bout en bout necessitant un vrai LLM.
"""

import csv
from pathlib import Path

import pytest

from eval.retrieval_eval import (
    GoldItem,
    QueryMetrics,
    candidate_recall,
    context_precision,
    context_recall,
    evaluate,
    evaluate_from_answer,
    load_gold_set,
    ndcg_at_k,
    summarize,
    to_markdown_table,
    to_sources,
    write_csv,
)
from src.domain.models import Answer

_FAKE_GOLD_YAML = """\
- id: q1
  question: "What fuels the propulsion system?"
  type: factual_specific
  relevant_sources:
    - propulsion.md
  key_points:
    - "Xenon fuels the ion thrusters."

- id: q2
  question: "What does the cafeteria serve and where are the safety exits?"
  type: multi_doc
  relevant_sources:
    - cafeteria.md
    - safety.md
"""


def _answer(retrieved: list[str], reranked: list[str], selected: list[str]) -> Answer:
    return Answer(
        text="reponse",
        sources=sorted(set(selected)),
        selected_chunk_ids=[],
        tokens_used=0,
        abstained=False,
        meta={
            "retrieved_sources": retrieved,
            "reranked_sources": reranked,
            "selected_sources": selected,
        },
    )


class TestToSources:
    def test_deduplicates_keeping_first_rank(self) -> None:
        assert to_sources(["a.md", "b.md", "a.md", "c.md"]) == ["a.md", "b.md", "c.md"]

    def test_empty_list(self) -> None:
        assert to_sources([]) == []


class TestCandidateRecall:
    def test_finds_all(self) -> None:
        assert candidate_recall(["file02", "file01", "file03"], {"file01", "file03"}) == pytest.approx(1.0)

    def test_finds_half(self) -> None:
        assert candidate_recall(["file01"], {"file01", "file02"}) == pytest.approx(0.5)

    def test_empty_gold_is_zero(self) -> None:
        assert candidate_recall(["file01"], set()) == 0.0

    def test_deduplicates_before_counting(self) -> None:
        # meme source retrouvee via 3 chunks : ne doit compter qu'une fois.
        assert candidate_recall(["file01", "file01", "file01"], {"file01", "file02"}) == pytest.approx(0.5)


class TestContextRecall:
    def test_same_formula_as_candidate_recall_on_selected_stage(self) -> None:
        assert context_recall(["file01", "file03"], {"file01", "file03"}) == pytest.approx(1.0)

    def test_empty_gold_is_zero(self) -> None:
        assert context_recall(["file01"], set()) == 0.0


class TestContextPrecision:
    def test_all_selected_relevant(self) -> None:
        assert context_precision(["file01", "file03"], {"file01", "file03"}) == pytest.approx(1.0)

    def test_half_selected_relevant(self) -> None:
        assert context_precision(["file01", "file02"], {"file01"}) == pytest.approx(0.5)

    def test_no_cap_from_gold_size_unlike_old_precision_at_k(self) -> None:
        # contrairement a l'ancien precision_at_k (plafonne a |gold|/k), 1.0
        # est atteignable meme si |selected| > |gold|, tant que tout est pertinent.
        assert context_precision(["file01", "file01b"], {"file01", "file01b", "file01c"}) == pytest.approx(1.0)

    def test_empty_selected_is_zero(self) -> None:
        assert context_precision([], {"file01"}) == 0.0

    def test_deduplicates_before_dividing(self) -> None:
        # sans deduplication, un doc redondant dans "selected" diluerait a tort
        # le denominateur ; avec to_sources(), 2x "file01" ne compte qu'une fois.
        assert context_precision(["file01", "file01", "file02"], {"file01"}) == pytest.approx(0.5)


class TestNdcgAtK:
    def test_matches_hand_computed_value(self) -> None:
        import math

        retrieved = ["file02", "file01", "file03"]
        relevant = {"file01", "file03"}
        expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 / math.log2(2) + 1 / math.log2(3))
        assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(expected)
        assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(0.6934264036172708)

    def test_perfect_ranking_is_one(self) -> None:
        retrieved = ["file01", "file03", "file02"]
        relevant = {"file01", "file03"}
        assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)

    def test_no_relevant_sources_is_zero(self) -> None:
        assert ndcg_at_k(["file01"], set(), k=1) == 0.0

    def test_duplicate_chunks_from_same_source_never_exceed_one(self) -> None:
        """Regression du bug reel (nDCG > 1.0 observe avec `chunking_markdown`,
        qui produit plusieurs chunks par document) : sans deduplication par
        source AVANT le calcul du DCG, un document sur-represente dans le
        classement (plusieurs chunks du meme fichier dans le top-k) comptait
        plusieurs fois au numerateur, alors que l'IDCG ne compte qu'un seul
        hit ideal par SOURCE distincte - le ratio pouvait alors depasser 1.0."""
        # 2 chunks de "boot_sequence.md" dans le top-3, 1 chunk non pertinent.
        reranked = ["boot_sequence.md", "boot_sequence.md", "other.md"]
        gold = {"boot_sequence.md"}

        score = ndcg_at_k(reranked, gold, k=3)

        assert score == pytest.approx(1.0)  # jamais > 1.0 (assert interne aussi)

    def test_asserts_bounds(self) -> None:
        # aucune entree ne devrait pouvoir violer l'assertion interne ; ce test
        # documente juste que l'assertion existe et ne se declenche pas ici.
        assert 0.0 <= ndcg_at_k(["a.md"], {"a.md"}, k=1) <= 1.0 + 1e-9


class TestLoadGoldSet:
    def test_parses_real_schema(self, tmp_path: Path) -> None:
        path = tmp_path / "gold.yaml"
        path.write_text(_FAKE_GOLD_YAML, encoding="utf-8")

        items = load_gold_set(path)

        assert len(items) == 2
        assert items[0] == GoldItem(
            id="q1",
            question="What fuels the propulsion system?",
            relevant_sources=["propulsion.md"],
            type="factual_specific",
        )
        assert items[1].relevant_sources == ["cafeteria.md", "safety.md"]

    def test_type_is_optional(self, tmp_path: Path) -> None:
        path = tmp_path / "gold.yaml"
        path.write_text(
            "- id: q1\n  question: \"q\"\n  relevant_sources: [a.md]\n", encoding="utf-8"
        )
        items = load_gold_set(path)
        assert items[0].type is None

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "gold.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            load_gold_set(path)

    def test_missing_relevant_sources_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "gold.yaml"
        path.write_text("- id: q1\n  question: \"q\"\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_gold_set(path)

    def test_empty_relevant_sources_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "gold.yaml"
        path.write_text(
            "- id: q1\n  question: \"q\"\n  relevant_sources: []\n", encoding="utf-8"
        )
        with pytest.raises(ValueError):
            load_gold_set(path)


class TestEvaluateFromAnswer:
    """Construit directement des `Answer` (pas de pipeline reel necessaire) :
    verifie le cablage entre `answer.meta` et les 4 metriques."""

    def test_all_stages_perfect(self) -> None:
        item = GoldItem(id="q1", question="q", relevant_sources=["a.md"])
        answer = _answer(retrieved=["a.md", "b.md"], reranked=["a.md", "b.md"], selected=["a.md"])

        metrics = evaluate_from_answer(answer, item, k=10)

        assert metrics.candidate_recall == pytest.approx(1.0)
        assert metrics.context_recall == pytest.approx(1.0)
        assert metrics.context_precision == pytest.approx(1.0)
        assert metrics.ndcg_at_10 == pytest.approx(1.0)

    def test_candidate_recall_guard_catches_source_missing_before_rerank(self) -> None:
        """`candidate_recall` < 1.0 signale qu'une source n'a jamais atteint le
        reranker : aucune amelioration du selector ne peut alors la recuperer."""
        item = GoldItem(id="q1", question="q", relevant_sources=["a.md", "missing.md"])
        answer = _answer(retrieved=["a.md"], reranked=["a.md"], selected=["a.md"])

        metrics = evaluate_from_answer(answer, item, k=10)

        assert metrics.candidate_recall == pytest.approx(0.5)
        assert metrics.context_recall == pytest.approx(0.5)

    def test_missing_meta_keys_treated_as_empty(self) -> None:
        item = GoldItem(id="q1", question="q", relevant_sources=["a.md"])
        answer = Answer(
            text="Information non trouvée dans les documents.",
            sources=[],
            selected_chunk_ids=[],
            tokens_used=0,
            abstained=True,
            meta={"stage": "guardrail_input"},
        )

        metrics = evaluate_from_answer(answer, item, k=10)

        assert metrics.candidate_recall == 0.0
        assert metrics.context_recall == 0.0
        assert metrics.context_precision == 0.0
        assert metrics.ndcg_at_10 == 0.0


class TestEvaluateRealPipeline:
    """Un seul test de bout en bout avec un vrai pipeline (BM25 + Ollama) :
    verifie le cablage complet, pas la logique des metriques (deja testee)."""

    def test_evaluate_runs_all_gold_items_and_reads_pipeline_meta(self, tmp_path: Path) -> None:
        from src.adapters.guardrails.basic import BasicGuardrail
        from src.adapters.llm.ollama import build_prompt, create_llm
        from src.adapters.llm.token_counter import TiktokenTokenCounter
        from src.adapters.reranking.noop import NoOpReranker
        from src.adapters.retrieval.bm25 import BM25Retriever
        from src.domain.budget import TopKSelector
        from src.domain.models import Chunk
        from src.domain.pipeline import Pipeline, PipelineConfig
        import tiktoken

        corpus = [
            Chunk(
                id="c1",
                text="The propulsion system uses xenon as fuel for the ion thrusters.",
                source="propulsion.md",
                n_tokens=12,
            ),
        ]
        pipeline = Pipeline(
            guardrail_input=BasicGuardrail(max_length=2000),
            retriever=BM25Retriever(chunks=corpus, k1=1.5, b=0.75),
            reranker=NoOpReranker(),
            selector=TopKSelector(),
            llm=create_llm(
                {"model": "qwen3:1.7b", "temperature": 0.1}, system_prompt="Answer from context.",
                max_output_tokens=100,
            ),
            guardrail_output=BasicGuardrail(max_length=4000),
            token_counter=TiktokenTokenCounter(tiktoken.get_encoding("cl100k_base")),
            assemble_prompt=build_prompt,
            system_prompt="Answer from context.",
            config=PipelineConfig(retrieve_k=5, rerank_top_n=5, budget_total=1024, reserve_answer=200),
        )
        gold_set = [GoldItem(id="q1", question="What fuel does the propulsion system use?", relevant_sources=["propulsion.md"])]

        results = evaluate(pipeline, gold_set, k=10)

        assert [r.id for r in results] == ["q1"]
        assert all(isinstance(r, QueryMetrics) for r in results)
        assert results[0].candidate_recall == pytest.approx(1.0)


class TestSummarize:
    def test_averages_across_queries(self) -> None:
        results = [
            QueryMetrics(id="q1", question="a", candidate_recall=1.0, context_recall=1.0, context_precision=1.0, ndcg_at_10=1.0),
            QueryMetrics(id="q2", question="b", candidate_recall=0.0, context_recall=0.0, context_precision=0.0, ndcg_at_10=0.0),
        ]
        summary = summarize(results)
        assert summary == {
            "candidate_recall": 0.5,
            "context_recall": 0.5,
            "context_precision": 0.5,
            "ndcg_at_10": 0.5,
        }

    def test_empty_results(self) -> None:
        summary = summarize([])
        assert summary == {
            "candidate_recall": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0,
            "ndcg_at_10": 0.0,
        }


class TestReports:
    _RESULTS = [
        QueryMetrics(id="q1", question="a", candidate_recall=1.0, context_recall=1.0, context_precision=1.0, ndcg_at_10=1.0),
        QueryMetrics(id="q2", question="b", candidate_recall=0.5, context_recall=0.5, context_precision=0.5, ndcg_at_10=0.5),
    ]
    _SUMMARY = {"candidate_recall": 0.75, "context_recall": 0.75, "context_precision": 0.75, "ndcg_at_10": 0.75}

    def test_markdown_table_contains_rows_and_mean(self) -> None:
        table = to_markdown_table(self._RESULTS, summary=self._SUMMARY)

        assert "| q1 |" in table
        assert "| q2 |" in table
        assert "Moyenne" in table
        assert "nDCG@10" in table

    def test_write_csv_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "results.csv"
        write_csv(self._RESULTS, summary=self._SUMMARY, path=path)

        with path.open(encoding="utf-8") as f:
            rows = list(csv.reader(f))

        assert rows[0] == ["question_id", "candidate_recall", "context_recall", "context_precision", "ndcg_at_10"]
        assert rows[1][0] == "q1"
        assert rows[2][0] == "q2"
        assert rows[3][0] == "MEAN"
        assert rows[3][1] == "0.75"
