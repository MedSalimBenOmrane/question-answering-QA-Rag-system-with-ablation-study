"""Tests des adapters de chunking (fixed, recursive, markdown, factory).

Utilise le vrai `TiktokenTokenCounter` (encodage cl100k_base reel) fourni par
la fixture `real_token_counter` : aucun compteur de tokens simule.
"""

import pytest

from src.adapters.chunking.factory import create_chunker
from src.adapters.chunking.fixed import FixedSizeChunker
from src.adapters.chunking.markdown import MarkdownChunker
from src.adapters.chunking.recursive import RecursiveChunker
from src.adapters.llm.token_counter import TiktokenTokenCounter
from src.domain.models import Chunk

MULTI_SOURCE_TEXT = """\
<!-- SOURCE: file01 — Ambiguous Uses of "Safe Mode" -->
# Ambiguous Uses of Safe Mode

The term Safe Mode is used in different contexts across the fleet manuals
and technician notes, causing confusion during emergency procedures.

<!-- SOURCE: file02 — Boot Sequence -->
# Boot Sequence

ZentroSoft initializes modules in a fixed order during boot: core services,
sensors, the AI layer, the UI layer, then the external network bridge.
"""


def _assert_common_invariants(chunks: list[Chunk], expected_sources: set[str]) -> None:
    assert len(chunks) > 0
    for c in chunks:
        assert c.text.strip() != ""
        assert c.source in expected_sources
        assert c.n_tokens > 0


class TestFixedSizeChunker:
    def test_chunks_multi_source_text(self, real_token_counter: TiktokenTokenCounter) -> None:
        chunker = FixedSizeChunker(
            token_counter=real_token_counter, chunk_size_tokens=15, overlap_tokens=3
        )
        chunks = chunker.chunk([("corpus.md", MULTI_SOURCE_TEXT)])
        _assert_common_invariants(chunks, {"file01", "file02"})

    def test_no_marker_falls_back_to_given_source(
        self, real_token_counter: TiktokenTokenCounter
    ) -> None:
        chunker = FixedSizeChunker(
            token_counter=real_token_counter, chunk_size_tokens=5, overlap_tokens=1
        )
        chunks = chunker.chunk([("plain.md", "one two three four five six seven eight")])
        _assert_common_invariants(chunks, {"plain.md"})

    def test_rejects_invalid_overlap(self, real_token_counter: TiktokenTokenCounter) -> None:
        with pytest.raises(ValueError):
            FixedSizeChunker(
                token_counter=real_token_counter, chunk_size_tokens=10, overlap_tokens=10
            )


class TestRecursiveChunker:
    def test_chunks_multi_source_text(self, real_token_counter: TiktokenTokenCounter) -> None:
        chunker = RecursiveChunker(
            token_counter=real_token_counter, chunk_size_tokens=15, overlap_tokens=3
        )
        chunks = chunker.chunk([("corpus.md", MULTI_SOURCE_TEXT)])
        _assert_common_invariants(chunks, {"file01", "file02"})

    def test_small_text_is_single_chunk(self, real_token_counter: TiktokenTokenCounter) -> None:
        chunker = RecursiveChunker(token_counter=real_token_counter, chunk_size_tokens=100)
        chunks = chunker.chunk([("plain.md", "a short piece of text")])
        assert len(chunks) == 1
        assert chunks[0].source == "plain.md"


class TestMarkdownChunker:
    def test_groups_by_header_and_keeps_source(
        self, real_token_counter: TiktokenTokenCounter
    ) -> None:
        chunker = MarkdownChunker(token_counter=real_token_counter)
        chunks = chunker.chunk([("corpus.md", MULTI_SOURCE_TEXT)])
        _assert_common_invariants(chunks, {"file01", "file02"})
        # une section groupee par source : un seul chunk par document ici
        assert len(chunks) == 2
        by_source = {c.source: c for c in chunks}
        assert by_source["file01"].text.startswith("# Ambiguous Uses of Safe Mode")
        assert by_source["file02"].text.startswith("# Boot Sequence")


class TestChunkerFactory:
    def test_creates_fixed_from_config(self, real_token_counter: TiktokenTokenCounter) -> None:
        chunker = create_chunker(
            {"strategy": "fixed", "fixed": {"chunk_size_tokens": 50, "overlap_tokens": 5}},
            real_token_counter,
        )
        assert isinstance(chunker, FixedSizeChunker)

    def test_creates_recursive_from_config(
        self, real_token_counter: TiktokenTokenCounter
    ) -> None:
        chunker = create_chunker(
            {"strategy": "recursive", "recursive": {"chunk_size_tokens": 50}},
            real_token_counter,
        )
        assert isinstance(chunker, RecursiveChunker)

    def test_creates_markdown_from_config(self, real_token_counter: TiktokenTokenCounter) -> None:
        chunker = create_chunker({"strategy": "markdown"}, real_token_counter)
        assert isinstance(chunker, MarkdownChunker)

    def test_unknown_strategy_raises(self, real_token_counter: TiktokenTokenCounter) -> None:
        with pytest.raises(ValueError):
            create_chunker({"strategy": "does-not-exist"}, real_token_counter)

    def test_missing_strategy_key_raises(self, real_token_counter: TiktokenTokenCounter) -> None:
        with pytest.raises(KeyError):
            create_chunker({}, real_token_counter)
