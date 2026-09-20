"""Tests de eval/run_ablation.py : merge de config, detection de reindexation,
rapports, et UNE execution reelle bout en bout minimale (retrieval + generation).

Aucun mock. Le test d'integration complet est reduit a 1 question et 1 config
sans reindexation, pour limiter le cout/temps des appels API reels (Claude).
Saute automatiquement si ANTHROPIC_API_KEY n'est pas configuree.
"""

import csv
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from eval.run_ablation import (
    ExperimentResult,
    changed_keys,
    deep_merge,
    discover_experiment_configs,
    load_experiment_override,
    requires_reindex,
    run_experiment,
    to_markdown_table,
    write_csv,
)

load_dotenv()
_requires_claude = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY non configuree : juge Claude indisponible",
)


class TestDeepMerge:
    def test_overrides_leaf_value(self) -> None:
        base = {"llm": {"model": "qwen3:1.7b", "temperature": 0.1}}
        result = deep_merge(base, {"llm": {"model": "smollm2:1.7b"}})
        assert result == {"llm": {"model": "smollm2:1.7b", "temperature": 0.1}}

    def test_does_not_mutate_base(self) -> None:
        base = {"llm": {"model": "qwen3:1.7b"}}
        deep_merge(base, {"llm": {"model": "smollm2:1.7b"}})
        assert base == {"llm": {"model": "qwen3:1.7b"}}

    def test_merges_multiple_nested_keys(self) -> None:
        base = {"retrieval": {"hybrid": True, "strategy": "dense", "rrf_k": 60}}
        result = deep_merge(base, {"retrieval": {"hybrid": False, "strategy": "bm25"}})
        assert result == {"retrieval": {"hybrid": False, "strategy": "bm25", "rrf_k": 60}}

    def test_empty_override_returns_copy_of_base(self) -> None:
        base = {"a": {"b": 1}}
        result = deep_merge(base, {})
        assert result == base
        assert result is not base


class TestChangedKeys:
    def test_single_key(self) -> None:
        assert changed_keys({"llm": {"model": "x"}}) == ["llm.model"]

    def test_multiple_keys_sorted(self) -> None:
        assert changed_keys({"retrieval": {"strategy": "bm25", "hybrid": False}}) == [
            "retrieval.hybrid",
            "retrieval.strategy",
        ]

    def test_empty_override(self) -> None:
        assert changed_keys({}) == []


class TestRequiresReindex:
    def test_chunking_change_requires_reindex(self) -> None:
        assert requires_reindex({"chunking": {"strategy": "recursive"}}) is True

    def test_embedding_change_requires_reindex(self) -> None:
        assert requires_reindex({"embedding": {"provider": "multilingual_e5"}}) is True

    def test_llm_change_does_not_require_reindex(self) -> None:
        assert requires_reindex({"llm": {"model": "smollm2:1.7b"}}) is False

    def test_empty_override_does_not_require_reindex(self) -> None:
        assert requires_reindex({}) is False


class TestDiscoverExperimentConfigs:
    def test_finds_all_yaml_files(self, tmp_path: Path) -> None:
        (tmp_path / "b.yaml").write_text("b: 1", encoding="utf-8")
        (tmp_path / "a.yaml").write_text("a: 1", encoding="utf-8")
        (tmp_path / "not_yaml.txt").write_text("x", encoding="utf-8")

        paths = discover_experiment_configs(tmp_path)

        assert [p.name for p in paths] == ["a.yaml", "b.yaml"]

    def test_finds_real_experiment_configs(self) -> None:
        paths = discover_experiment_configs()
        names = {p.stem for p in paths}
        assert {"llm_smollm", "llm_qwen3", "llm_qwen35", "chunking_recursive", "selection_topk"} <= names


class TestLoadExperimentOverride:
    def test_loads_partial_override(self, tmp_path: Path) -> None:
        path = tmp_path / "x.yaml"
        path.write_text("llm:\n  model: smollm2:1.7b\n", encoding="utf-8")
        assert load_experiment_override(path) == {"llm": {"model": "smollm2:1.7b"}}

    def test_real_experiment_files_change_only_documented_keys(self) -> None:
        for path in discover_experiment_configs():
            override = load_experiment_override(path)
            # chaque fichier doit toucher au moins une cle, et rester "petit"
            # (1 ou 2 cles liees, jamais un override massif involontaire).
            assert 1 <= len(changed_keys(override)) <= 2, path.name


class TestExperimentResultCompositeScore:
    def test_averages_all_metrics(self) -> None:
        result = ExperimentResult(
            name="x",
            changed_keys=[],
            retrieval_metrics={"precision_at_k": 1.0, "recall_at_k": 1.0},
            generation_metrics={"faithfulness_custom": 0.0, "relevancy_custom": 0.0},
        )
        assert result.composite_score == pytest.approx(0.5)


class TestReports:
    _RESULTS = [
        ExperimentResult(
            name="worse",
            changed_keys=["llm.model"],
            retrieval_metrics={"precision_at_k": 0.5, "recall_at_k": 0.5, "mrr": 0.5, "ndcg_at_k": 0.5},
            generation_metrics={
                "faithfulness_custom": 0.5,
                "relevancy_custom": 0.5,
                "faithfulness_ragas": 0.5,
                "answer_relevancy_ragas": 0.5,
                "context_precision_ragas": 0.5,
                "context_recall_ragas": 0.5,
            },
        ),
        ExperimentResult(
            name="better",
            changed_keys=[],
            retrieval_metrics={"precision_at_k": 1.0, "recall_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0},
            generation_metrics={
                "faithfulness_custom": 1.0,
                "relevancy_custom": 1.0,
                "faithfulness_ragas": 1.0,
                "answer_relevancy_ragas": 1.0,
                "context_precision_ragas": 1.0,
                "context_recall_ragas": 1.0,
            },
        ),
    ]

    def test_markdown_table_sorted_best_first(self) -> None:
        table = to_markdown_table(self._RESULTS)
        assert table.index("| better |") < table.index("| worse |")

    def test_baseline_label_for_empty_changed_keys(self) -> None:
        table = to_markdown_table(self._RESULTS)
        assert "(baseline)" in table

    def test_write_csv_sorted_best_first(self, tmp_path: Path) -> None:
        path = tmp_path / "results.csv"
        write_csv(self._RESULTS, path)

        with path.open(encoding="utf-8") as f:
            rows = list(csv.reader(f))

        assert rows[1][0] == "better"
        assert rows[2][0] == "worse"


@_requires_claude
class TestRunExperimentReal:
    """Une seule execution reelle bout en bout (retrieval + generation),
    1 question, config sans reindexation (reutilise le corpus deja indexe
    par les tests d'autres briques via la fixture ci-dessous)."""

    def test_run_experiment_end_to_end_minimal(self, tmp_path: Path) -> None:
        from eval.generation_eval import GenerationCase, load_ragas_judge_and_embeddings, load_judge_llm
        from eval.retrieval_eval import GoldItem
        from src.cli import run_index

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "doc1.md").write_text(
            "<!-- SOURCE: file01 -->\n# Propulsion\n\n"
            "The propulsion system uses xenon as fuel for the ion thrusters.\n",
            encoding="utf-8",
        )

        base_config = {
            "chunking": {"strategy": "fixed", "fixed": {"chunk_size_tokens": 200, "overlap_tokens": 20}},
            "embedding": {
                "provider": "bge_m3",
                "bge_m3": {"model_name": "BAAI/bge-m3", "device": "cpu", "batch_size": 32},
            },
            "llm": {"model": "qwen3:1.7b", "temperature": 0.1},
            "vectorstore": {
                "persist_directory": str(tmp_path / "chroma"),
                "collection_name": "test_ablation",
                "chunks_path": str(tmp_path / "chunks.json"),
            },
            "retrieval": {"hybrid": False, "strategy": "bm25", "bm25": {"k1": 1.5, "b": 0.75}, "rrf_k": 60},
            "reranking": {"enabled": False},
            "selection": {"strategy": "topk"},
            "guardrails": {"input": {"max_length": 2000}, "output": {"max_length": 4000}},
            "pipeline": {"retrieve_k": 3, "rerank_top_n": 3},
            "budget": {"total": 1024, "reserve_answer": 200},
            "token_counter": {"provider": "tiktoken", "tiktoken": {"encoding": "cl100k_base"}},
        }
        run_index(base_config, docs_dir=docs_dir)

        system_prompt = (
            "Answer only from context. If absent, say exactly: "
            "\"Information non trouvée dans les documents.\" Cite [Source: fileNN]."
        )
        gold_retrieval_items = [GoldItem(id="q1", question="xenon fuel", relevant_sources=["file01"])]
        gold_generation_cases = [
            GenerationCase(id="q1", question="What fuel does the propulsion system use?", reference="Xenon.")
        ]

        judge_llm_raw = load_judge_llm()
        ragas_llm, ragas_embeddings = load_ragas_judge_and_embeddings(base_config["embedding"])

        result = run_experiment(
            name="selection_topk_test",
            override={"selection": {"strategy": "topk"}},
            base_config=base_config,
            gold_retrieval_items=gold_retrieval_items,
            gold_generation_cases=gold_generation_cases,
            system_prompt=system_prompt,
            judge_llm_raw=judge_llm_raw,
            ragas_llm=ragas_llm,
            ragas_embeddings=ragas_embeddings,
        )

        assert result.name == "selection_topk_test"
        assert result.changed_keys == ["selection.strategy"]
        assert 0.0 <= result.retrieval_metrics["recall_at_k"] <= 1.0
        assert 0.0 <= result.generation_metrics["faithfulness_ragas"] <= 1.0
        assert 0.0 <= result.composite_score <= 1.0
