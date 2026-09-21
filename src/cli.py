"""CLI : `index` (construit l'index vectoriel) et `ask "<question>"` (interroge le pipeline).

Usage:
    uv run python -m src.cli index
    uv run python -m src.cli ask "Quelle est la procedure de securite ?"
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from src.adapters.chunking.factory import create_chunker
from src.adapters.embedding.factory import create_embedder
from src.adapters.llm.token_counter import create_token_counter
from src.adapters.vectorstore.chroma import create_vectorstore
from src.application.answer import answer_question, build_pipeline
from src.domain.models import Answer, Chunk
from src.observability.logging import RequestLogger

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "default.yaml"
_SYSTEM_PROMPT_PATH = _REPO_ROOT / "src" / "prompts" / "system.txt"
_DOCS_DIR = _REPO_ROOT / "data" / "docs"


def load_config(path: Path = _DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Charge une configuration YAML (deja au format attendu par les factories)."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_system_prompt(path: Path = _SYSTEM_PROMPT_PATH) -> str:
    """Charge le contenu du system prompt."""
    return path.read_text(encoding="utf-8")


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    """Persiste la liste des chunks indexes en JSON (necessaire au retriever BM25)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(chunk) for chunk in chunks]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunks(path: Path | str) -> list[Chunk]:
    """Recharge la liste des chunks indexes depuis le JSON persiste par `save_chunks`."""
    if isinstance(path, str):
        path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**item) for item in payload]


def run_index(config: dict[str, Any] | None = None, docs_dir: Path = _DOCS_DIR) -> list[Chunk]:
    """Indexe le corpus : chunk, embed, persiste dans le VectorStore et sur disque.

    Etape offline (cf. CLAUDE.md) : separee de la reponse aux questions.

    Args:
        config: Configuration complete (par defaut : `config/default.yaml`).
        docs_dir: Dossier contenant les documents `.md` a indexer.

    Returns:
        La liste des chunks indexes.

    Raises:
        FileNotFoundError: Si `docs_dir` ne contient aucun fichier `.md`.
    """
    config = config if config is not None else load_config()

    doc_paths = sorted(docs_dir.glob("*.md"))
    if not doc_paths:
        raise FileNotFoundError(f"aucun fichier .md trouve dans {docs_dir}")

    docs = [(path.name, path.read_text(encoding="utf-8")) for path in doc_paths]

    token_counter = create_token_counter(config["token_counter"])
    chunker = create_chunker(config["chunking"], token_counter)
    chunks = chunker.chunk(docs)

    embedder = create_embedder(config["embedding"])
    vectors = embedder.embed([chunk.text for chunk in chunks])

    vectorstore = create_vectorstore(config["vectorstore"])
    vectorstore.add(chunks, vectors)

    save_chunks(chunks, Path(config["vectorstore"]["chunks_path"]))

    return chunks


def run_ask(question: str, config: dict[str, Any] | None = None) -> Answer:
    """Repond a une question via le pipeline complet.

    Args:
        question: La question posee par l'utilisateur.
        config: Configuration complete (par defaut : `config/default.yaml`).

    Returns:
        La reponse produite (`Answer`).

    Raises:
        FileNotFoundError: Si l'index (chunks persistes) n'existe pas encore
            (executer `index` avant `ask`).
    """
    config = config if config is not None else load_config()

    chunks_path = Path(config["vectorstore"]["chunks_path"])
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"index introuvable ({chunks_path}) : executer `index` avant `ask`"
        )
    chunks = load_chunks(chunks_path)

    system_prompt = load_system_prompt()
    pipeline = build_pipeline(config, chunks, system_prompt)
    logger = RequestLogger()

    return answer_question(question, pipeline, logger)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI (`index`, `ask "<question>"`)."""
    parser = argparse.ArgumentParser(prog="context-aware-qa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("index", help="Indexer le corpus (data/docs/)")

    ask_parser = subparsers.add_parser("ask", help="Poser une question")
    ask_parser.add_argument("question", type=str)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Point d'entree CLI : dispatch `index` / `ask`."""
    args = build_arg_parser().parse_args(argv)

    if args.command == "index":
        chunks = run_index()
        print(f"{len(chunks)} chunks indexes.")
    elif args.command == "ask":
        answer = run_ask(args.question)
        print(answer.text)
        if answer.sources:
            print(f"\nSources: {', '.join(answer.sources)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
