"""Phase 3 Optimization: Find best LLM with optimal retrieval + reranking.

Uses best retrieval and reranking configs from Phases 1-2.
Tests different LLM models for generation.
Evaluates full pipeline with LLM-as-judge (faithfulness + answer_correctness).
"""

import csv
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment variables for Bedrock judge
load_dotenv()

from eval.generation_eval import (
    evaluate_answer,
    load_gold_cases,
    load_judge_llm,
    judge_model_name,
)
from eval.retrieval_eval import evaluate_from_answer, load_gold_set
from src.application.answer import build_pipeline
from src.cli import load_config, load_system_prompt, load_chunks, save_chunks
from src.adapters.chunking.factory import create_chunker
from src.adapters.embedding.factory import create_embedder
from src.adapters.llm.token_counter import create_token_counter
from src.adapters.vectorstore.chroma import create_vectorstore

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PHASE1_RESULTS = _REPO_ROOT / "eval" / "phase1_retrieval_optimization.csv"
_PHASE2_RESULTS = _REPO_ROOT / "eval" / "phase2_reranking_optimization.csv"
_RESULTS_CSV = _REPO_ROOT / "eval" / "phase3_generation_optimization.csv"
_DOCS_DIR = _REPO_ROOT / "data" / "docs"

# LLM models to test (Ollama models - lightweight only: 1.7B-2B)
LLM_MODELS = [
    "qwen3:1.7b",      # Qwen 3 baseline (1.7B)
    "qwen3.5:2b",      # Qwen 3.5 improved (2B)
    "smollm2:1.7b",    # SmolLM 2 optimized (1.7B)
]


def load_best_pipeline_config() -> dict[str, Any]:
    """Load best retrieval + reranking config from Phases 1-2."""
    # Load Phase 1 results
    if not _PHASE1_RESULTS.exists():
        raise FileNotFoundError(
            f"Phase 1 results not found: {_PHASE1_RESULTS}\n"
            "Run optimize_retrieval.py first!"
        )

    with _PHASE1_RESULTS.open("r", encoding="utf-8") as f:
        import csv as csv_module
        reader = csv_module.DictReader(f)
        phase1_best = next(reader)

    # Load Phase 2 results
    if not _PHASE2_RESULTS.exists():
        raise FileNotFoundError(
            f"Phase 2 results not found: {_PHASE2_RESULTS}\n"
            "Run optimize_reranking.py first!"
        )

    with _PHASE2_RESULTS.open("r", encoding="utf-8") as f:
        reader = csv_module.DictReader(f)
        phase2_best = next(reader)

    print(f"Using optimal pipeline config from Phases 1-2:")
    print(f"  - Chunking: {phase1_best['chunking']}")
    print(f"  - Embedding: {phase1_best['embedding']}")
    print(f"  - Retrieval: {phase1_best['retrieval']}")
    print(f"  - Reranking: {phase2_best['name']}")
    print()

    return {
        "chunking": phase1_best["chunking"],
        "embedding": phase1_best["embedding"],
        "retrieval": phase1_best["retrieval"],
        "reranking": phase2_best["name"],
    }


def create_config_with_llm(
    base_config: dict[str, Any],
    best_pipeline: dict[str, Any],
    llm_model: str,
) -> dict[str, Any]:
    """Create config with best pipeline + specific LLM."""
    config = base_config.copy()

    # Apply best retrieval
    config["chunking"]["strategy"] = best_pipeline["chunking"]
    config["embedding"]["provider"] = best_pipeline["embedding"]

    if best_pipeline["retrieval"] == "hybrid":
        config["retrieval"]["hybrid"] = True
        config["retrieval"]["strategy"] = "dense"
    else:
        config["retrieval"]["hybrid"] = False
        config["retrieval"]["strategy"] = best_pipeline["retrieval"]

    # Apply best reranking
    if best_pipeline["reranking"] == "disabled":
        config["reranking"]["enabled"] = False
    elif best_pipeline["reranking"] == "cross_encoder":
        config["reranking"]["enabled"] = True
        config["reranking"]["model_name"] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        config["reranking"]["device"] = "cpu"
    elif best_pipeline["reranking"] == "llm_bedrock":
        config["reranking"]["enabled"] = True
        config["reranking"]["provider"] = "llm"
        config["reranking"]["model_id"] = "global.anthropic.claude-sonnet-4-6"
        config["reranking"]["region"] = "eu-west-3"

    # Apply LLM model
    config["llm"]["model"] = llm_model

    return config


def reindex_with_best_pipeline(config: dict[str, Any]) -> list:
    """Reindex corpus once with best retrieval + reranking config."""
    print("  → Reindexing with best pipeline config from Phases 1-2...")

    # Load documents
    doc_paths = sorted(_DOCS_DIR.glob("*.md"))
    docs = [(path.name, path.read_text(encoding="utf-8")) for path in doc_paths]

    # Chunk documents
    token_counter = create_token_counter(config["token_counter"])
    chunker = create_chunker(config["chunking"], token_counter)
    chunks = chunker.chunk(docs)

    # Embed chunks
    embedder = create_embedder(config["embedding"])
    vectors = embedder.embed([chunk.text for chunk in chunks])

    # Reset vectorstore
    vectorstore = create_vectorstore(config["vectorstore"])
    vectorstore.reset()
    vectorstore.add(chunks, vectors)

    # Save chunks
    save_chunks(chunks, Path(config["vectorstore"]["chunks_path"]))

    print(f"  ✓ Reindexed {len(chunks)} chunks")
    return chunks


def evaluate_config(
    config: dict[str, Any],
    gold_retrieval: list,
    gold_generation: list,
    system_prompt: str,
    judge,
    judge_model: str,
    chunks: list,
) -> dict[str, Any]:
    """Evaluate full pipeline with retrieval + generation metrics."""
    pipeline = build_pipeline(config, chunks, system_prompt)

    # Retrieval metrics
    retrieval_results = []
    for item in gold_retrieval:
        answer = pipeline.run(item.question)
        metrics = evaluate_from_answer(answer, item)
        retrieval_results.append(metrics)

    avg_retrieval = {
        "candidate_recall": sum(r.candidate_recall for r in retrieval_results) / len(retrieval_results),
        "context_recall": sum(r.context_recall for r in retrieval_results) / len(retrieval_results),
        "context_precision": sum(r.context_precision for r in retrieval_results) / len(retrieval_results),
        "ndcg_at_10": sum(r.ndcg_at_10 for r in retrieval_results) / len(retrieval_results),
    }

    # Generation metrics
    generation_results = []
    for case in gold_generation:
        answer = pipeline.run(case.question)
        gen_metrics = evaluate_answer(
            answer=answer,
            gold_case=case,
            judge=judge,
            judge_model=judge_model,
        )
        generation_results.append(gen_metrics)

    # Average generation metrics
    faithfulness_values = [r["faithfulness"] for r in generation_results if r["faithfulness"] is not None]
    answer_correctness_values = [r["answer_correctness"] for r in generation_results if r["answer_correctness"] is not None]

    avg_generation = {
        "faithfulness": sum(faithfulness_values) / len(faithfulness_values) if faithfulness_values else 0.0,
        "answer_correctness": sum(answer_correctness_values) / len(answer_correctness_values) if answer_correctness_values else 0.0,
        "n_claims_total": sum(r["n_claims_total"] for r in generation_results),
        "n_key_points_covered": sum(r["n_key_points_covered"] for r in generation_results),
        "n_key_points_total": sum(r["n_key_points_total"] for r in generation_results),
    }

    return {**avg_retrieval, **avg_generation}


def main() -> None:
    """Run Phase 3 optimization: LLM generation."""
    print("=" * 80)
    print("PHASE 3: LLM Generation Optimization")
    print("=" * 80)

    # Load best pipeline config
    best_pipeline = load_best_pipeline_config()

    # Load configs and judge
    base_config = load_config()
    system_prompt = load_system_prompt()
    gold_retrieval = load_gold_set()
    gold_generation = load_gold_cases()
    judge = load_judge_llm()
    judge_model = judge_model_name()

    # Reindex once with best pipeline config
    first_config = create_config_with_llm(base_config, best_pipeline, LLM_MODELS[0])
    chunks = reindex_with_best_pipeline(first_config)
    print()

    # Test each LLM
    results = []
    for i, llm_model in enumerate(LLM_MODELS, 1):
        print(f"[{i}/{len(LLM_MODELS)}] Testing LLM: {llm_model}")

        config = create_config_with_llm(base_config, best_pipeline, llm_model)

        try:
            metrics = evaluate_config(
                config,
                gold_retrieval,
                gold_generation,
                system_prompt,
                judge,
                judge_model,
                chunks,
            )

            results.append({
                "name": llm_model,
                **metrics,
            })

            print(f"  → Faithfulness: {metrics['faithfulness']:.3f}, "
                  f"Correctness: {metrics['answer_correctness']:.3f}, "
                  f"Precision: {metrics['context_precision']:.3f}")
        except Exception as e:
            print(f"  → ERROR: {e}")
            continue

        print()

    # Composite score (same as ablation study)
    for r in results:
        r["composite"] = (
            0.30 * r["context_precision"]
            + 0.15 * r["ndcg_at_10"]
            + 0.30 * r["faithfulness"]
            + 0.25 * r["answer_correctness"]
        )

    results_sorted = sorted(results, key=lambda x: x["composite"], reverse=True)

    # Write CSV
    _RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with _RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name",
            "candidate_recall", "context_recall", "context_precision", "ndcg_at_10",
            "faithfulness", "answer_correctness",
            "n_claims_total", "n_key_points_covered", "n_key_points_total",
            "composite"
        ])
        writer.writeheader()
        writer.writerows(results_sorted)

    # Print results
    print("=" * 80)
    print("LLM MODELS RANKED:")
    print("=" * 80)

    if not results_sorted:
        print("❌ No LLM completed successfully!")
        print("Check errors above for details.")
        sys.exit(1)

    for i, r in enumerate(results_sorted, 1):
        print(f"{i}. {r['name']}")
        print(f"   Composite: {r['composite']:.4f}")
        print(f"   Faithfulness: {r['faithfulness']:.3f}, "
              f"Correctness: {r['answer_correctness']:.3f}, "
              f"Precision: {r['context_precision']:.3f}")
        print()

    best = results_sorted[0]
    print("=" * 80)
    print("OPTIMAL CONFIGURATION:")
    print("=" * 80)
    print(f"Chunking: {best_pipeline['chunking']}")
    print(f"Embedding: {best_pipeline['embedding']}")
    print(f"Retrieval: {best_pipeline['retrieval']}")
    print(f"Reranking: {best_pipeline['reranking']}")
    print(f"LLM: {best['name']}")
    print(f"\nComposite Score: {best['composite']:.4f}")
    print(f"\nResults written to: {_RESULTS_CSV}")


if __name__ == "__main__":
    main()
