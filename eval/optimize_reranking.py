"""Phase 2 Optimization: Find best reranking strategy with optimal retrieval config.

Uses the best retrieval config from Phase 1 and tests different reranking strategies.
Evaluates retrieval metrics (context_precision affected by reranking).
"""

import csv
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load environment variables for Bedrock credentials
load_dotenv()

from eval.retrieval_eval import evaluate_from_answer, load_gold_set
from src.application.answer import build_pipeline
from src.cli import load_config, load_system_prompt, load_chunks, save_chunks
from src.adapters.chunking.factory import create_chunker
from src.adapters.embedding.factory import create_embedder
from src.adapters.llm.token_counter import create_token_counter
from src.adapters.vectorstore.chroma import create_vectorstore

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PHASE1_RESULTS = _REPO_ROOT / "eval" / "phase1_retrieval_optimization.csv"
_RESULTS_CSV = _REPO_ROOT / "eval" / "phase2_reranking_optimization.csv"
_DOCS_DIR = _REPO_ROOT / "data" / "docs"

# Reranking strategies to test
RERANKING_CONFIGS = [
    {"name": "disabled", "enabled": False},
    {
        "name": "cross_encoder",
        "enabled": True,
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "device": "cpu",
    },
    {
        "name": "llm_bedrock",
        "enabled": True,
        "provider": "llm",
        "model_id": "global.anthropic.claude-sonnet-4-6",
        "region": "eu-west-3",
    },
]


def load_best_retrieval_config() -> dict[str, Any]:
    """Load best retrieval config from Phase 1 results."""
    if not _PHASE1_RESULTS.exists():
        raise FileNotFoundError(
            f"Phase 1 results not found: {_PHASE1_RESULTS}\n"
            "Run optimize_retrieval.py first!"
        )

    with _PHASE1_RESULTS.open("r", encoding="utf-8") as f:
        import csv
        reader = csv.DictReader(f)
        best = next(reader)  # First row is best (sorted)

    print(f"Using best retrieval config from Phase 1:")
    print(f"  - Chunking: {best['chunking']}")
    print(f"  - Embedding: {best['embedding']}")
    print(f"  - Retrieval: {best['retrieval']}")
    print(f"  - Composite: {best['composite']}")
    print()

    return {
        "chunking": best["chunking"],
        "embedding": best["embedding"],
        "retrieval": best["retrieval"],
    }


def create_config_with_reranking(
    base_config: dict[str, Any],
    best_retrieval: dict[str, Any],
    reranking: dict[str, Any],
) -> dict[str, Any]:
    """Create config with best retrieval + specific reranking."""
    config = base_config.copy()

    # Apply best retrieval config
    config["chunking"]["strategy"] = best_retrieval["chunking"]
    config["embedding"]["provider"] = best_retrieval["embedding"]

    if best_retrieval["retrieval"] == "hybrid":
        config["retrieval"]["hybrid"] = True
        config["retrieval"]["strategy"] = "dense"
    else:
        config["retrieval"]["hybrid"] = False
        config["retrieval"]["strategy"] = best_retrieval["retrieval"]

    # Apply reranking config
    config["reranking"] = reranking.copy()
    config["reranking"].pop("name")  # Remove name key (not part of config)

    return config


def reindex_with_best_retrieval(config: dict[str, Any]) -> list:
    """Reindex corpus once with best retrieval config from Phase 1."""
    print("  → Reindexing with best retrieval config from Phase 1...")

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
    gold_items: list,
    system_prompt: str,
    chunks: list,
) -> dict[str, float]:
    """Evaluate config on retrieval metrics."""
    pipeline = build_pipeline(config, chunks, system_prompt)

    results = []
    for item in gold_items:
        answer = pipeline.run(item.question)
        metrics = evaluate_from_answer(answer, item)
        results.append(metrics)

    avg_metrics = {
        "candidate_recall": sum(r.candidate_recall for r in results) / len(results),
        "context_recall": sum(r.context_recall for r in results) / len(results),
        "context_precision": sum(r.context_precision for r in results) / len(results),
        "ndcg_at_10": sum(r.ndcg_at_10 for r in results) / len(results),
    }

    return avg_metrics


def main() -> None:
    """Run Phase 2 optimization: reranking."""
    print("=" * 80)
    print("PHASE 2: Reranking Optimization")
    print("=" * 80)

    # Load best retrieval config from Phase 1
    best_retrieval = load_best_retrieval_config()

    # Load base config
    base_config = load_config()
    system_prompt = load_system_prompt()
    gold_items = load_gold_set()

    # Reindex once with best retrieval config
    first_config = create_config_with_reranking(base_config, best_retrieval, RERANKING_CONFIGS[0])
    chunks = reindex_with_best_retrieval(first_config)
    print()

    # Test each reranking strategy
    results = []
    for i, reranking in enumerate(RERANKING_CONFIGS, 1):
        name = reranking["name"]
        print(f"[{i}/{len(RERANKING_CONFIGS)}] Testing reranking: {name}")

        config = create_config_with_reranking(base_config, best_retrieval, reranking)
        metrics = evaluate_config(config, gold_items, system_prompt, chunks)

        results.append({
            "name": name,
            **metrics,
        })

        print(f"  → Recall: {metrics['context_recall']:.3f}, "
              f"Precision: {metrics['context_precision']:.3f}, "
              f"nDCG: {metrics['ndcg_at_10']:.3f}")
        print()

    # Composite score (precision most important for reranking)
    for r in results:
        r["composite"] = (
            0.50 * r["context_precision"]  # Reranking affects precision most
            + 0.25 * r["context_recall"]
            + 0.25 * r["ndcg_at_10"]
        )

    results_sorted = sorted(results, key=lambda x: x["composite"], reverse=True)

    # Write CSV
    _RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with _RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "candidate_recall", "context_recall", "context_precision",
            "ndcg_at_10", "composite"
        ])
        writer.writeheader()
        writer.writerows(results_sorted)

    # Print results
    print("=" * 80)
    print("RERANKING STRATEGIES RANKED:")
    print("=" * 80)
    for i, r in enumerate(results_sorted, 1):
        print(f"{i}. {r['name']}")
        print(f"   Composite: {r['composite']:.4f}")
        print(f"   Precision: {r['context_precision']:.3f}, "
              f"Recall: {r['context_recall']:.3f}, "
              f"nDCG: {r['ndcg_at_10']:.3f}")
        print()

    best = results_sorted[0]
    print(f"Best reranking: {best['name']}")
    print(f"\nResults written to: {_RESULTS_CSV}")


if __name__ == "__main__":
    main()
