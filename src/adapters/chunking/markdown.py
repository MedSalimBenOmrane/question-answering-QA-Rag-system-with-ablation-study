"""Strategie de chunking par titres Markdown.

Adapter du port `Chunker`. Decoupe le texte a chaque ligne de titre Markdown
(`#`, `##`, ...) : chaque section (titre + contenu jusqu'au titre suivant)
forme un chunk unique et reste groupee (non re-decoupee).
"""

import re

from src.domain.models import Chunk
from src.domain.ports import TokenCounter
from src.adapters.chunking._shared import make_chunk, split_by_source_marker

_HEADER_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)


class MarkdownChunker:
    """Chunker qui decoupe un document Markdown en sections groupees par titre."""

    def __init__(self, token_counter: TokenCounter) -> None:
        """Initialise le chunker.

        Args:
            token_counter: Compteur de tokens injecte (contrat `TokenCounter`).
        """
        self._token_counter = token_counter

    def chunk(self, docs: list[tuple[str, str]]) -> list[Chunk]:
        """Decoupe les documents en sections Markdown groupees par titre.

        Args:
            docs: Liste de tuples (source, texte).

        Returns:
            La liste des chunks produits, avec `source` correctement renseigne
            meme si un fichier contient plusieurs documents marques `SOURCE`.
        """
        chunks: list[Chunk] = []
        for source, text in docs:
            for sub_source, sub_text in split_by_source_marker(source, text):
                chunks.extend(self._chunk_one(sub_source, sub_text))
        return chunks

    def _chunk_one(self, source: str, text: str) -> list[Chunk]:
        sections = self._split_sections(text)
        return [
            make_chunk(source, section, idx, self._token_counter)
            for idx, section in enumerate(sections)
            if section.strip()
        ]

    def _split_sections(self, text: str) -> list[str]:
        """Decoupe `text` en sections, chacune demarrant a un titre Markdown."""
        headers = list(_HEADER_RE.finditer(text))
        if not headers:
            return [text] if text.strip() else []

        sections: list[str] = []

        preamble = text[: headers[0].start()].strip()
        if preamble:
            sections.append(preamble)

        for idx, header in enumerate(headers):
            end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
            section = text[header.start() : end].strip()
            if section:
                sections.append(section)

        return sections
