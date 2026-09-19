"""Tests de eval/retrieval_eval.py : metriques, chargement du gold set, rapports.

Utilise un gold set FACTICE (pas eval/gold_retrieval.yaml, reserve au vrai
corpus rempli par l'utilisateur) et un retriever BM25 reel (rapide, aucun
modele lourd necessaire) sur un mini-corpus en memoire.
"""

import csv
import math
from pathlib import Path

import pytest
import yaml

from eval.retrieval_eval import (
    GoldItem,
    QueryMetrics,
    _ndcg_at_k,
    _precision_at_k,
    _reciprocal_rank,
    _recall_at_k,
    evaluate,
    evaluate_query,
    load_gold_set,
    summarize,
    to_markdown_table,
    write_csv,
)
from src.adapters.retrieval.bm25 import BM25Retriever
from src.domain.models import Chunk

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

_CORPUS = [
    Chunk(
        id="c1",
        text="The propulsion system uses xenon as fuel for the ion thrusters.",
        source="propulsion.md",
        n_tokens=12,
    ),
    Chunk(
        id="c2",
        text="The cafeteria menu today offers pasta and a green salad.",
        source="cafeteria.md",
        n_tokens=11,
    ),
    Chunk(
        id="c3",
        text="Safety procedures list all emergency exits located on deck two.",
        source="safety.md",
        n_tokens=11,
    ),
]


class TestMetricFunctions:
    """Valeurs verifiees a la main (cf. resume de reponse pour le detail du calcul)."""

    def test_precision_at_k(self) -> None:
        retrieved = ["file02", "file01", "file03"]
        relevant = {"file01", "file03"}
        assert _precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)

    def test_precision_at_k_zero_k(self) -> None:
        assert _precision_at_k(["file01"], {"file01"}, k=0) == 0.0

    def test_recall_at_k_finds_all(self) -> None:
        retrieved = ["file02", "file01", "file03"]
        relevant = {"file01", "file03"}
        assert _recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)

    def test_recall_at_k_finds_none(self) -> None:
        retrieved = ["file04", "file05"]
        relevant = {"file01"}
        assert _recall_at_k(retrieved, relevant, k=2) == 0.0

    def test_recall_at_k_empty_relevant_set(self) -> None:
        assert _recall_at_k(["file01"], set(), k=1) == 0.0

    def test_reciprocal_rank_hit_at_rank_2(self) -> None:
        retrieved = ["file02", "file01", "file03"]
        relevant = {"file01", "file03"}
        assert _reciprocal_rank(retrieved, relevant, k=3) == pytest.approx(0.5)

    def test_reciprocal_rank_no_hit(self) -> None:
        assert _reciprocal_rank(["file02"], {"file01"}, k=1) == 0.0

    def test_reciprocal_rank_hit_outside_k_is_ignored(self) -> None:
        retrieved = ["file02", "file01"]
        relevant = {"file01"}
        assert _reciprocal_rank(retrieved, relevant, k=1) == 0.0

    def test_ndcg_at_k_matches_hand_computed_value(self) -> None:
        retrieved = ["file02", "file01", "file03"]
        relevant = {"file01", "file03"}
        # dcg = 0/log2(2) + 1/log2(3) + 1/log2(4) ; idcg = 1/log2(2) + 1/log2(3)
        expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 / math.log2(2) + 1 / math.log2(3))
        assert _ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(expected)
        assert _ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(0.6934264036172708)

    def test_ndcg_at_k_perfect_ranking_is_one(self) -> None:
        retrieved = ["file01", "file03", "file02"]
        relevant = {"file01", "file03"}
        assert _ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)

    def test_ndcg_at_k_no_relevant_sources_is_zero(self) -> None:
        assert _ndcg_at_k(["file01"], set(), k=1) == 0.0


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


class TestEvaluateWithRealRetriever:
    def test_evaluate_query_perfect_match_on_lexical_query(self) -> None:
        retriever = BM25Retriever(chunks=_CORPUS, k1=1.5, b=0.75)
        item = GoldItem(
            id="q1",
            question="xenon fuel ion thrusters",
            relevant_sources=["propulsion.md"],
        )

        metrics = evaluate_query(retriever, item, k=3)

        assert metrics.reciprocal_rank == pytest.approx(1.0)
        assert metrics.recall_at_k == pytest.approx(1.0)

    def test_evaluate_runs_all_gold_items_in_order(self, tmp_path: Path) -> None:
        path = tmp_path / "gold.yaml"
        path.write_text(_FAKE_GOLD_YAML, encoding="utf-8")
        gold_set = load_gold_set(path)
        retriever = BM25Retriever(chunks=_CORPUS, k1=1.5, b=0.75)

        results = evaluate(retriever, gold_set, k=3)

        assert [r.id for r in results] == ["q1", "q2"]
        assert all(isinstance(r, QueryMetrics) for r in results)


class TestSummarize:
    def test_averages_across_queries(self) -> None:
        results = [
            QueryMetrics(id="q1", question="a", precision_at_k=1.0, recall_at_k=1.0, reciprocal_rank=1.0, ndcg_at_k=1.0),
            QueryMetrics(id="q2", question="b", precision_at_k=0.0, recall_at_k=0.0, reciprocal_rank=0.0, ndcg_at_k=0.0),
        ]
        summary = summarize(results)
        assert summary == {
            "precision_at_k": 0.5,
            "recall_at_k": 0.5,
            "mrr": 0.5,
            "ndcg_at_k": 0.5,
        }

    def test_empty_results(self) -> None:
        summary = summarize([])
        assert summary == {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
        }


class TestReports:
    _RESULTS = [
        QueryMetrics(id="q1", question="a", precision_at_k=1.0, recall_at_k=1.0, reciprocal_rank=1.0, ndcg_at_k=1.0),
        QueryMetrics(id="q2", question="b", precision_at_k=0.5, recall_at_k=0.5, reciprocal_rank=0.5, ndcg_at_k=0.5),
    ]
    _SUMMARY = {"precision_at_k": 0.75, "recall_at_k": 0.75, "mrr": 0.75, "ndcg_at_k": 0.75}

    def test_markdown_table_contains_rows_and_mean(self) -> None:
        table = to_markdown_table(self._RESULTS, k=3, summary=self._SUMMARY)

        assert "| q1 |" in table
        assert "| q2 |" in table
        assert "Moyenne" in table
        assert "Precision@3" in table

    def test_write_csv_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "results.csv"
        write_csv(self._RESULTS, k=3, summary=self._SUMMARY, path=path)

        with path.open(encoding="utf-8") as f:
            rows = list(csv.reader(f))

        assert rows[0] == ["question_id", "precision_at_3", "recall_at_3", "rr", "ndcg_at_3"]
        assert rows[1][0] == "q1"
        assert rows[2][0] == "q2"
        assert rows[3][0] == "MEAN"
        assert rows[3][1] == "0.75"
