"""Run full 3-phase optimization pipeline and configure default.yaml with best config.

Phase 1: Optimize retrieval pipeline (chunking + embedding + retrieval)
Phase 2: Optimize reranking (with best retrieval fixed)
Phase 3: Optimize LLM generation (with best retrieval + reranking fixed)

Final: Apply best configuration to config/default.yaml
"""

import csv
import subprocess
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PHASE1_CSV = _REPO_ROOT / "eval" / "phase1_retrieval_optimization.csv"
_PHASE2_CSV = _REPO_ROOT / "eval" / "phase2_reranking_optimization.csv"
_PHASE3_CSV = _REPO_ROOT / "eval" / "phase3_generation_optimization.csv"
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "default.yaml"


def run_phase(phase_num: int, script_name: str) -> bool:
    """Run optimization phase script."""
    print()
    print("=" * 80)
    print(f"PHASE {phase_num}: {script_name}")
    print("=" * 80)
    print()

    result = subprocess.run(
        [sys.executable, "-m", f"eval.{script_name}"],
        cwd=_REPO_ROOT,
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"\n❌ Phase {phase_num} failed!")
        return False

    print(f"\n✅ Phase {phase_num} completed successfully!")
    return True


def load_best_config() -> dict:
    """Load best configuration from all phases."""
    # Phase 1
    with _PHASE1_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        phase1_best = next(reader)

    # Phase 2
    with _PHASE2_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        phase2_best = next(reader)

    # Phase 3
    with _PHASE3_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        phase3_best = next(reader)

    return {
        "chunking": phase1_best["chunking"],
        "embedding": phase1_best["embedding"],
        "retrieval": phase1_best["retrieval"],
        "reranking": phase2_best["name"],
        "llm": phase3_best["name"],
        "composite": float(phase3_best["composite"]),
        "metrics": {
            "context_precision": float(phase3_best["context_precision"]),
            "context_recall": float(phase3_best["context_recall"]),
            "faithfulness": float(phase3_best["faithfulness"]),
            "answer_correctness": float(phase3_best["answer_correctness"]),
        }
    }


def apply_config_to_default(best_config: dict) -> None:
    """Apply best configuration to config/default.yaml."""
    # Load current config
    with _DEFAULT_CONFIG.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Update chunking
    config["chunking"]["strategy"] = best_config["chunking"]

    # Update embedding
    config["embedding"]["provider"] = best_config["embedding"]

    # Update retrieval
    if best_config["retrieval"] == "hybrid":
        config["retrieval"]["hybrid"] = True
        config["retrieval"]["strategy"] = "dense"
    else:
        config["retrieval"]["hybrid"] = False
        config["retrieval"]["strategy"] = best_config["retrieval"]

    # Update reranking
    if best_config["reranking"] == "disabled":
        config["reranking"] = {"enabled": False}
    elif best_config["reranking"] == "cross_encoder":
        config["reranking"] = {
            "enabled": True,
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "device": "cpu",
        }
    elif best_config["reranking"] == "llm_bedrock":
        config["reranking"] = {
            "enabled": True,
            "provider": "llm",
            "model_id": "global.anthropic.claude-sonnet-4-6",
            "region": "eu-west-3",
        }

    # Update LLM
    config["llm"]["model"] = best_config["llm"]

    # Write back
    with _DEFAULT_CONFIG.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Updated {_DEFAULT_CONFIG} with optimal configuration")


def print_final_report(best_config: dict) -> None:
    """Print final optimization report."""
    print()
    print("=" * 80)
    print("🎯 OPTIMIZATION COMPLETE - FINAL REPORT")
    print("=" * 80)
    print()
    print("OPTIMAL CONFIGURATION:")
    print("-" * 80)
    print(f"  Chunking:   {best_config['chunking']}")
    print(f"  Embedding:  {best_config['embedding']}")
    print(f"  Retrieval:  {best_config['retrieval']}")
    print(f"  Reranking:  {best_config['reranking']}")
    print(f"  LLM:        {best_config['llm']}")
    print()
    print("PERFORMANCE METRICS:")
    print("-" * 80)
    print(f"  Composite Score:     {best_config['composite']:.4f}")
    print(f"  Context Precision:   {best_config['metrics']['context_precision']:.3f}")
    print(f"  Context Recall:      {best_config['metrics']['context_recall']:.3f}")
    print(f"  Faithfulness:        {best_config['metrics']['faithfulness']:.3f}")
    print(f"  Answer Correctness:  {best_config['metrics']['answer_correctness']:.3f}")
    print()
    print("NEXT STEPS:")
    print("-" * 80)
    print("  1. Test on UI: uv run streamlit run src/ui/streamlit_app.py")
    print("  2. Review detailed results:")
    print(f"     - Phase 1: {_PHASE1_CSV}")
    print(f"     - Phase 2: {_PHASE2_CSV}")
    print(f"     - Phase 3: {_PHASE3_CSV}")
    print("=" * 80)


def main() -> None:
    """Run full 3-phase optimization."""
    print("=" * 80)
    print("🚀 STARTING FULL PIPELINE OPTIMIZATION")
    print("=" * 80)
    print()
    print("This will run 3 phases sequentially:")
    print("  Phase 1: Retrieval Pipeline (chunking + embedding + retrieval)")
    print("  Phase 2: Reranking (with best retrieval)")
    print("  Phase 3: LLM Generation (with best retrieval + reranking)")
    print()

    # Check existing phases
    phase1_exists = _PHASE1_CSV.exists()
    phase2_exists = _PHASE2_CSV.exists()
    phase3_exists = _PHASE3_CSV.exists()

    if phase1_exists:
        print(f"✓ Phase 1 results found: {_PHASE1_CSV}")
    if phase2_exists:
        print(f"✓ Phase 2 results found: {_PHASE2_CSV}")
    if phase3_exists:
        print(f"✓ Phase 3 results found: {_PHASE3_CSV}")

    if phase1_exists or phase2_exists or phase3_exists:
        print()
        skip = input("Skip completed phases? (Y/n): ").strip().lower()
        if skip != 'n':
            print("→ Skipping completed phases")
        else:
            phase1_exists = phase2_exists = phase3_exists = False
            print("→ Re-running all phases")

    print()
    remaining_time = 0
    if not phase1_exists:
        remaining_time += 20
    if not phase2_exists:
        remaining_time += 15
    if not phase3_exists:
        remaining_time += 30
    print(f"Estimated time: {remaining_time} minutes")
    print()

    input("Press Enter to continue or Ctrl+C to cancel...")

    # Run Phase 1
    if not phase1_exists:
        if not run_phase(1, "optimize_retrieval"):
            sys.exit(1)
    else:
        print()
        print("=" * 80)
        print("PHASE 1: optimize_retrieval")
        print("=" * 80)
        print("→ Skipped (results already exist)")

    # Run Phase 2
    if not phase2_exists:
        if not run_phase(2, "optimize_reranking"):
            sys.exit(1)
    else:
        print()
        print("=" * 80)
        print("PHASE 2: optimize_reranking")
        print("=" * 80)
        print("→ Skipped (results already exist)")

    # Run Phase 3
    if not phase3_exists:
        if not run_phase(3, "optimize_generation"):
            sys.exit(1)
    else:
        print()
        print("=" * 80)
        print("PHASE 3: optimize_generation")
        print("=" * 80)
        print("→ Skipped (results already exist)")

    # Load best config
    best_config = load_best_config()

    # Apply to default.yaml
    apply_config_to_default(best_config)

    # Print final report
    print_final_report(best_config)


if __name__ == "__main__":
    main()
