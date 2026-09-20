"""Tests de eval/run_ablation.py : merge de config, refus multi-sections,
determinisme (repetitions/composite/garde-fou), rapports, et UNE execution
reelle bout en bout minimale (1 question, 1 repetition).

Aucun mock. Le test d'integration complet est reduit au minimum pour limiter
le cout/temps des appels API reels (Claude). Saute automatiquement si
ANTHROPIC_API_KEY n'est pas configuree.
"""

import csv
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from eval.run_ablation import (
    ExperimentResult,
    RepetitionResult,
    changed_keys,
    config_hash,
    deep_merge,
    discover_experiment_configs,
    load_experiment_override,
    refuse_if_multi_section,
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


class TestRefuseIfMultiSection:
    """Interpretation retenue pour "plus d'une cle" (etape 4) : compte les
    SECTIONS de premier niveau, pas les chemins pointes complets."""

    def test_single_section_two_keys_is_allowed(self) -> None:
        # retrieval.hybrid + retrieval.strategy : 2 chemins, 1 seule section.
        refuse_if_multi_section("x", {"retrieval": {"hybrid": False, "strategy": "bm25"}})

    def test_single_key_is_allowed(self) -> None:
        refuse_if_multi_section("x", {"llm": {"model": "smollm2:1.7b"}})

    def test_empty_override_is_allowed(self) -> None:
        refuse_if_multi_section("baseline", {})

    def test_two_sections_raises(self) -> None:
        with pytest.raises(ValueError, match="2 sections"):
            refuse_if_multi_section(
                "bad", {"retrieval": {"hybrid": False}, "selection": {"strategy": "topk"}}
            )

    def test_real_experiment_files_touch_a_single_section(self) -> None:
        for path in discover_experiment_configs():
            override = load_experiment_override(path)
            refuse_if_multi_section(path.stem, override)  # ne doit jamais lever


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


class TestConfigHash:
    def test_deterministic_for_same_config(self) -> None:
        config = {"a": 1, "b": {"c": 2}}
        assert config_hash(config) == config_hash({"b": {"c": 2}, "a": 1})  # ordre des cles indifferent

    def test_differs_for_different_config(self) -> None:
        assert config_hash({"a": 1}) != config_hash({"a": 2})

    def test_is_short_hex(self) -> None:
        h = config_hash({"a": 1})
        assert len(h) == 12
        int(h, 16)  # leve si pas hexadecimal


def _repetition(
    context_recall: float = 1.0,
    faithfulness: float | None = 0.8,
    answer_correctness: float | None = 0.8,
) -> RepetitionResult:
    return RepetitionResult(
        retrieval_metrics={
            "candidate_recall": 1.0,
            "context_recall": context_recall,
            "context_precision": 0.9,
            "ndcg_at_10": 0.95,
        },
        faithfulness=faithfulness,
        n_claims_total=5,
        answer_correctness=answer_correctness,
        n_key_points_covered=4,
        n_key_points_total=5,
        uncovered_key_point_ids=["q1-kp5"] if answer_correctness is not None and answer_correctness < 1.0 else [],
    )


class TestExperimentResultMetricMeanStd:
    def test_mean_and_std_across_repetitions(self) -> None:
        result = ExperimentResult(
            name="x", changed_keys=[], config_hash="abc", judge_model="m", reranker_version="r",
            repetitions=[_repetition(faithfulness=0.6), _repetition(faithfulness=1.0)],
        )
        mean, std = result.metric_mean_std("faithfulness")
        assert mean == pytest.approx(0.8)
        assert std == pytest.approx(0.2)

    def test_retrieval_metric_is_identical_across_repetitions_std_zero(self) -> None:
        result = ExperimentResult(
            name="x", changed_keys=[], config_hash="abc", judge_model="m", reranker_version="r",
            repetitions=[_repetition(), _repetition(), _repetition()],
        )
        mean, std = result.metric_mean_std("context_recall")
        assert mean == pytest.approx(1.0)
        assert std == pytest.approx(0.0)

    def test_none_values_excluded_from_mean(self) -> None:
        result = ExperimentResult(
            name="x", changed_keys=[], config_hash="abc", judge_model="m", reranker_version="r",
            repetitions=[_repetition(faithfulness=None), _repetition(faithfulness=1.0)],
        )
        mean, _ = result.metric_mean_std("faithfulness")
        assert mean == pytest.approx(1.0)  # seule la valeur non-None compte


class TestExperimentResultComposite:
    def test_computes_weighted_average_when_valid(self) -> None:
        # _repetition() fixe context_precision=0.9 et ndcg_at_10=0.95 :
        # composite = 0.30*0.9 + 0.15*0.95 + 0.30*1.0 + 0.25*1.0 = 0.9625.
        result = ExperimentResult(
            name="x", changed_keys=[], config_hash="abc", judge_model="m", reranker_version="r",
            repetitions=[_repetition(context_recall=1.0, faithfulness=1.0, answer_correctness=1.0)],
        )
        assert result.composite == pytest.approx(0.30 * 0.9 + 0.15 * 0.95 + 0.30 * 1.0 + 0.25 * 1.0)

    def test_none_when_context_recall_below_threshold(self) -> None:
        """Garde-fou obligatoire : recall < 0.98 -> run INVALIDE, jamais un score bas."""
        result = ExperimentResult(
            name="x", changed_keys=[], config_hash="abc", judge_model="m", reranker_version="r",
            repetitions=[_repetition(context_recall=0.5, faithfulness=1.0, answer_correctness=1.0)],
        )
        assert result.composite is None

    def test_zero_when_total_abstention(self) -> None:
        result = ExperimentResult(
            name="x", changed_keys=[], config_hash="abc", judge_model="m", reranker_version="r",
            repetitions=[_repetition(context_recall=1.0, faithfulness=None, answer_correctness=None)],
        )
        assert result.composite == 0.0

    def test_never_masks_bad_recall_with_good_other_metrics(self) -> None:
        """Regression du bug reel : `reranking_disabled` (context_recall RAGAS
        0.13 dans l'ancien harness) finissait devant llm_smollm au score
        composite via une simple moyenne ponderee. Impossible desormais."""
        bad_recall_but_otherwise_perfect = ExperimentResult(
            name="bad", changed_keys=[], config_hash="a", judge_model="m", reranker_version="r",
            repetitions=[_repetition(context_recall=0.13, faithfulness=1.0, answer_correctness=1.0)],
        )
        assert bad_recall_but_otherwise_perfect.composite is None


class TestAllUncoveredKeyPointIds:
    def test_deduplicates_across_repetitions(self) -> None:
        result = ExperimentResult(
            name="x", changed_keys=[], config_hash="abc", judge_model="m", reranker_version="r",
            repetitions=[_repetition(answer_correctness=0.8), _repetition(answer_correctness=0.8)],
        )
        assert result.all_uncovered_key_point_ids == ["q1-kp5"]


class TestReports:
    _RESULTS = [
        ExperimentResult(
            name="worse", changed_keys=["llm.model"], config_hash="a1", judge_model="m", reranker_version="r",
            repetitions=[_repetition(faithfulness=0.5, answer_correctness=0.5)],
        ),
        ExperimentResult(
            name="better", changed_keys=[], config_hash="a2", judge_model="m", reranker_version="r",
            repetitions=[_repetition(faithfulness=1.0, answer_correctness=1.0)],
        ),
        ExperimentResult(
            name="invalid", changed_keys=["reranking.enabled"], config_hash="a3", judge_model="m", reranker_version="r",
            repetitions=[_repetition(context_recall=0.1, faithfulness=1.0, answer_correctness=1.0)],
        ),
    ]

    def test_markdown_table_sorted_best_first_invalid_last(self) -> None:
        table = to_markdown_table(self._RESULTS)
        assert table.index("| better |") < table.index("| worse |") < table.index("| invalid |")
        assert "INVALIDE" in table

    def test_baseline_label_for_empty_changed_keys(self) -> None:
        table = to_markdown_table(self._RESULTS)
        assert "(baseline)" in table

    def test_write_csv_sorted_best_first(self, tmp_path: Path) -> None:
        path = tmp_path / "results.csv"
        write_csv(self._RESULTS, path)

        with path.open(encoding="utf-8") as f:
            rows = list(csv.reader(f))

        assert rows[0][0] == "name"
        assert rows[1][0] == "better"
        assert rows[2][0] == "worse"
        assert rows[3][0] == "invalid"
        assert rows[3][-3] == ""  # composite vide (run invalide)

    def test_write_csv_header_matches_row_order(self, tmp_path: Path) -> None:
        path = tmp_path / "results.csv"
        write_csv(self._RESULTS, path)

        with path.open(encoding="utf-8") as f:
            rows = list(csv.reader(f))

        header = rows[0]
        assert header.index("faithfulness_mean") < header.index("n_claims_total")
        assert header.index("answer_correctness_mean") < header.index("n_claims_total")
        row = rows[1]
        # colonnes numeriques : convertibles en float sans lever (verifie
        # que l'ordre ecrit correspond bien a l'ordre annonce par l'entete).
        float(row[header.index("faithfulness_mean")])
        float(row[header.index("answer_correctness_mean")])


@_requires_claude
class TestRunExperimentReal:
    """Une seule execution reelle bout en bout (retrieval + generation + juge),
    1 question, 1 repetition, config sans reindexation."""

    def test_run_experiment_end_to_end_minimal(self, tmp_path: Path) -> None:
        from eval.generation_eval import GenerationCase, load_judge_llm, judge_model_name
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
        gold_retrieval_items = [GoldItem(id="q1", question="What fuel does the propulsion system use?", relevant_sources=["file01"])]
        gold_generation_cases = [
            GenerationCase(
                id="q1",
                question="What fuel does the propulsion system use?",
                key_points=["Xenon fuels the ion thrusters."],
            )
        ]

        judge = load_judge_llm()

        result = run_experiment(
            name="selection_topk_test",
            override={"selection": {"strategy": "topk"}},
            base_config=base_config,
            gold_retrieval_items=gold_retrieval_items,
            gold_generation_cases=gold_generation_cases,
            system_prompt=system_prompt,
            judge=judge,
            judge_model=judge_model_name(),
            n_repetitions=1,
        )

        assert result.name == "selection_topk_test"
        assert result.changed_keys == ["selection.strategy"]
        assert len(result.repetitions) == 1
        context_recall_mean, _ = result.metric_mean_std("context_recall")
        assert 0.0 <= context_recall_mean <= 1.0
        faithfulness_mean, _ = result.metric_mean_std("faithfulness")
        assert 0.0 <= faithfulness_mean <= 1.0
        assert result.composite is None or 0.0 <= result.composite <= 1.0
