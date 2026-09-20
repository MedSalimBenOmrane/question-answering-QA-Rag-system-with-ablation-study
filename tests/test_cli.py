"""Test d'integration bout en bout : `index` puis `ask` sur un mini-corpus.

Aucun mock : vrai chunking, vrai embedder BGE-M3, vrai Chroma, vrai
retrieval hybride (dense + BM25 fusionnes par RRF), vrai reranking
cross-encoder, vraie selection (config/default.yaml), vrai serveur Ollama
(qwen3:1.7b) - c'est la config par defaut du projet (config/default.yaml)
utilisee telle quelle ; seuls les chemins de persistance sont rediriges
vers un dossier temporaire pour isoler le test du vrai data/chroma.
"""

from pathlib import Path
from typing import Any

import pytest

from src.cli import build_arg_parser, load_config, run_ask, run_index

_FILE01 = """<!-- SOURCE: file01 — Propulsion -->
# Propulsion

The propulsion system uses xenon as fuel for the ion thrusters. It provides
efficient long-duration thrust for deep space maneuvers.
"""

_FILE02 = """<!-- SOURCE: file02 — Cafeteria -->
# Cafeteria

The cafeteria menu today offers pasta and a green salad. Dinner service
starts at 18:00.
"""


def _isolated_config(tmp_path: Path) -> dict[str, Any]:
    config = load_config()
    config["vectorstore"] = dict(config["vectorstore"])
    config["vectorstore"]["persist_directory"] = str(tmp_path / "chroma")
    config["vectorstore"]["chunks_path"] = str(tmp_path / "chunks.json")
    return config


def _write_mini_corpus(docs_dir: Path) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "doc1.md").write_text(_FILE01, encoding="utf-8")
    (docs_dir / "doc2.md").write_text(_FILE02, encoding="utf-8")


class TestEndToEndIndexThenAsk:
    def test_index_then_ask_grounded_question(self, tmp_path: Path) -> None:
        config = _isolated_config(tmp_path)
        docs_dir = tmp_path / "docs"
        _write_mini_corpus(docs_dir)

        chunks = run_index(config, docs_dir=docs_dir)

        assert len(chunks) > 0
        assert {c.source for c in chunks} == {"file01", "file02"}
        assert Path(config["vectorstore"]["chunks_path"]).exists()

        answer = run_ask("What fuel does the propulsion system use?", config)

        assert answer.abstained is False
        assert "xenon" in answer.text.lower()
        assert "file01" in answer.sources
        assert answer.tokens_used > 0
        assert len(answer.selected_chunk_ids) > 0
        assert answer.meta["context_budget"] > 0

    def test_ask_abstains_on_unanswerable_question(self, tmp_path: Path) -> None:
        config = _isolated_config(tmp_path)
        docs_dir = tmp_path / "docs"
        _write_mini_corpus(docs_dir)

        run_index(config, docs_dir=docs_dir)
        answer = run_ask("What is the capital of France?", config)

        assert answer.abstained is True

    def test_ask_before_index_raises(self, tmp_path: Path) -> None:
        config = _isolated_config(tmp_path)

        with pytest.raises(FileNotFoundError):
            run_ask("anything", config)

    def test_index_with_no_docs_raises(self, tmp_path: Path) -> None:
        config = _isolated_config(tmp_path)
        empty_dir = tmp_path / "empty_docs"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            run_index(config, docs_dir=empty_dir)


class TestCLIArgParsing:
    """Verifie le dispatch du vrai parser CLI sans executer le pipeline complet."""

    def test_index_command_parses(self) -> None:
        args = build_arg_parser().parse_args(["index"])
        assert args.command == "index"

    def test_ask_command_parses_question(self) -> None:
        args = build_arg_parser().parse_args(["ask", "What fuel does it use?"])
        assert args.command == "ask"
        assert args.question == "What fuel does it use?"

    def test_missing_command_raises_systemexit(self) -> None:
        with pytest.raises(SystemExit):
            build_arg_parser().parse_args([])
