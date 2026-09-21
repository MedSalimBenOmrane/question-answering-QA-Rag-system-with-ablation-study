"""Phase 1 Optimization: Find best retrieval pipeline (chunking + embedding + retrieval).

Evaluates all combinations of chunking, embedding, and retrieval strategies
to find the optimal configuration for the retrieval pipeline.
Only evaluates retrieval metrics (no LLM generation/judge).
"""

import csv
import itertools
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from eval.retrieval_eval import evaluate_from_answer, load_gold_set
from src.application.answer import build_pipeline
from src.cli import load_config, load_system_prompt, load_chunks, save_chunks
from src.adapters.chunking.factory import create_chunker
from src.adapters.embedding.factory import create_embedder
from src.adapters.llm.token_counter import create_token_counter
from src.adapters.vectorstore.chroma import create_vectorstore

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_CSV = _REPO_ROOT / "eval" / "phase1_retrieval_optimization.csv"
_DOCS_DIR = _REPO_ROOT / "data" / "docs"

# Cache for reindexing
_LAST_EMBEDDING_PROVIDER = None
_LAST_CHUNKING_STRATEGY = None
_CACHED_CHUNKS = None

# Grid search space
CHUNKING_STRATEGIES = ["fixed", "recursive", "markdown"]
EMBEDDING_PROVIDERS = ["bge_m3", "multilingual_e5"]
RETRIEVAL_STRATEGIES = [
    {"hybrid": True, "strategy": "dense"},  # baseline hybrid
    {"hybrid": False, "strategy": "dense"},  # dense only
    {"hybrid": False, "strategy": "bm25"},   # bm25 only
]


def create_config_variant(
    base_config: dict[str, Any],
    chunking: str,
    embedding: str,
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    """Create config variant by modifying chunking, embedding, and retrieval."""
    config = base_config.copy()

    # Chunking
    config["chunking"]["strategy"] = chunking

    # Embedding
    config["embedding"]["provider"] = embedding

    # Retrieval
    config["retrieval"]["hybrid"] = retrieval["hybrid"]
    config["retrieval"]["strategy"] = retrieval["strategy"]

    return config


def config_name(chunking: str, embedding: str, retrieval: dict[str, Any]) -> str:
    """Generate human-readable config name."""
    ret_name = "hybrid" if retrieval["hybrid"] else retrieval["strategy"]
    return f"{chunking}_{embedding}_{ret_name}"


def reindex_if_needed(config: dict[str, Any]) -> list:
    """Reindex corpus if chunking or embedding changed."""
    global _LAST_EMBEDDING_PROVIDER, _LAST_CHUNKING_STRATEGY, _CACHED_CHUNKS

    current_embedding = config["embedding"]["provider"]
    current_chunking = config["chunking"]["strategy"]

    # Check if reindex needed
    if (
        _LAST_EMBEDDING_PROVIDER == current_embedding
        and _LAST_CHUNKING_STRATEGY == current_chunking
        and _CACHED_CHUNKS is not None
    ):
        # No change, reuse cached chunks
        return _CACHED_CHUNKS

    print(f"  → Reindexing with chunking={current_chunking}, embedding={current_embedding}")

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

    # Reset vectorstore (delete and recreate collection)
    vectorstore = create_vectorstore(config["vectorstore"])
    vectorstore.reset()
    vectorstore.add(chunks, vectors)

    # Save chunks to disk
    save_chunks(chunks, Path(config["vectorstore"]["chunks_path"]))

    # Update cache
    _LAST_EMBEDDING_PROVIDER = current_embedding
    _LAST_CHUNKING_STRATEGY = current_chunking
    _CACHED_CHUNKS = chunks

    return chunks


def evaluate_config(
    config: dict[str, Any],
    gold_items: list,
    system_prompt: str,
) -> dict[str, float]:
    """Evaluate single config on retrieval metrics only."""
    # Reindex if embedding or chunking changed
    chunks = reindex_if_needed(config)

    # Build pipeline
    pipeline = build_pipeline(config, chunks, system_prompt)

    # Evaluate on all questions
    results = []
    for item in gold_items:
        answer = pipeline.run(item.question)
        metrics = evaluate_from_answer(answer, item)
        results.append(metrics)

    # Average across questions
    avg_metrics = {
        "candidate_recall": sum(r.candidate_recall for r in results) / len(results),
        "context_recall": sum(r.context_recall for r in results) / len(results),
        "context_precision": sum(r.context_precision for r in results) / len(results),
        "ndcg_at_10": sum(r.ndcg_at_10 for r in results) / len(results),
    }

    return avg_metrics


def main() -> None:
    """Run Phase 1 optimization: retrieval pipeline."""
    print("=" * 80)
    print("PHASE 1: Retrieval Pipeline Optimization")
    print("=" * 80)
    print(f"Chunking strategies: {len(CHUNKING_STRATEGIES)}")
    print(f"Embedding providers: {len(EMBEDDING_PROVIDERS)}")
    print(f"Retrieval strategies: {len(RETRIEVAL_STRATEGIES)}")
    print(f"Total combinations: {len(CHUNKING_STRATEGIES) * len(EMBEDDING_PROVIDERS) * len(RETRIEVAL_STRATEGIES)}")
    print()

    # Load base config and gold set
    base_config = load_config()
    system_prompt = load_system_prompt()
    gold_items = load_gold_set()

    # Grid search
    results = []
    total = len(CHUNKING_STRATEGIES) * len(EMBEDDING_PROVIDERS) * len(RETRIEVAL_STRATEGIES)
    counter = 0

    for chunking, embedding, retrieval in itertools.product(
        CHUNKING_STRATEGIES, EMBEDDING_PROVIDERS, RETRIEVAL_STRATEGIES
    ):
        counter += 1
        name = config_name(chunking, embedding, retrieval)
        print(f"[{counter}/{total}] Evaluating: {name}")

        # Create config variant
        config = create_config_variant(base_config, chunking, embedding, retrieval)

        # Evaluate
        metrics = evaluate_config(config, gold_items, system_prompt)

        # Store results
        results.append({
            "name": name,
            "chunking": chunking,
            "embedding": embedding,
            "retrieval": "hybrid" if retrieval["hybrid"] else retrieval["strategy"],
            **metrics,
        })

        print(f"  → Recall: {metrics['context_recall']:.3f}, "
              f"Precision: {metrics['context_precision']:.3f}, "
              f"nDCG: {metrics['ndcg_at_10']:.3f}")
        print()

    # Sort by composite score (weighted: 40% precision, 30% recall, 30% ndcg)
    for r in results:
        r["composite"] = (
            0.40 * r["context_precision"]
            + 0.30 * r["context_recall"]
            + 0.30 * r["ndcg_at_10"]
        )

    results_sorted = sorted(results, key=lambda x: x["composite"], reverse=True)

    # Write CSV
    _RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with _RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "chunking", "embedding", "retrieval",
            "candidate_recall", "context_recall", "context_precision", "ndcg_at_10",
            "composite"
        ])
        writer.writeheader()
        writer.writerows(results_sorted)

    # Print top 5
    print("=" * 80)
    print("TOP 5 RETRIEVAL CONFIGURATIONS:")
    print("=" * 80)
    for i, r in enumerate(results_sorted[:5], 1):
        print(f"{i}. {r['name']}")
        print(f"   Composite: {r['composite']:.4f}")
        print(f"   Precision: {r['context_precision']:.3f}, "
              f"Recall: {r['context_recall']:.3f}, "
              f"nDCG: {r['ndcg_at_10']:.3f}")
        print()

    best = results_sorted[0]
    print(f"Best configuration saved: {best['name']}")
    print(f"  - Chunking: {best['chunking']}")
    print(f"  - Embedding: {best['embedding']}")
    print(f"  - Retrieval: {best['retrieval']}")
    print(f"\nResults written to: {_RESULTS_CSV}")


if __name__ == "__main__":
    main()
