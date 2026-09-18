"""Strategie de chunking recursive (par separateurs hierarchiques) avec overlap.

Adapter du port `Chunker`. Essaie de decouper le texte sur des separateurs de
plus en plus fins (paragraphe, ligne, phrase, mot) jusqu'a respecter le budget
de tokens (mesure via le `TokenCounter` injecte), afin de preserver au mieux
la structure semantique du texte.
"""

from src.domain.models import Chunk
from src.domain.ports import TokenCounter
from src.adapters.chunking._shared import make_chunk, split_by_source_marker

_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")


class RecursiveChunker:
    """Chunker recursif par separateurs hierarchiques, avec chevauchement."""

    def __init__(
        self,
        token_counter: TokenCounter,
        chunk_size_tokens: int,
        overlap_tokens: int = 0,
    ) -> None:
        """Initialise le chunker.

        Args:
            token_counter: Compteur de tokens injecte (contrat `TokenCounter`).
            chunk_size_tokens: Taille maximale d'un chunk, en tokens.
            overlap_tokens: Nombre de tokens partages entre deux chunks consecutifs.

        Raises:
            ValueError: Si `chunk_size_tokens` <= 0 ou `overlap_tokens` >= `chunk_size_tokens`.
        """
        if chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens doit etre strictement positif")
        if overlap_tokens < 0 or overlap_tokens >= chunk_size_tokens:
            raise ValueError("overlap_tokens doit etre dans [0, chunk_size_tokens[")

        self._token_counter = token_counter
        self._chunk_size_tokens = chunk_size_tokens
        self._overlap_tokens = overlap_tokens

    def chunk(self, docs: list[tuple[str, str]]) -> list[Chunk]:
        """Decoupe les documents recursivement en respectant le budget de tokens.

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
        if not text.strip():
            return []

        pieces = self._split_recursive(text.strip(), list(_SEPARATORS))
        pieces = self._apply_overlap(pieces)

        return [
            make_chunk(source, piece, idx, self._token_counter)
            for idx, piece in enumerate(pieces)
            if piece.strip()
        ]

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Decoupe `text` en morceaux respectant le budget de tokens."""
        if self._token_counter.count(text) <= self._chunk_size_tokens:
            return [text]

        if not separators:
            mid = len(text) // 2
            return self._split_recursive(text[:mid], []) + self._split_recursive(text[mid:], [])

        sep, *rest = separators
        parts = [p for p in text.split(sep) if p]
        if len(parts) <= 1:
            return self._split_recursive(text, rest)

        pieces: list[str] = []
        current = ""
        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if self._token_counter.count(candidate) <= self._chunk_size_tokens:
                current = candidate
                continue
            if current:
                pieces.append(current)
            if self._token_counter.count(part) <= self._chunk_size_tokens:
                current = part
            else:
                pieces.extend(self._split_recursive(part, rest))
                current = ""
        if current:
            pieces.append(current)
        return pieces

    def _apply_overlap(self, pieces: list[str]) -> list[str]:
        """Ajoute, en tete de chaque piece, la fin de la piece precedente."""
        if self._overlap_tokens <= 0 or len(pieces) < 2:
            return pieces

        result = [pieces[0]]
        for previous, current in zip(pieces, pieces[1:]):
            tail_words = self._overlap_tail(previous.split())
            merged = f"{' '.join(tail_words)} {current}" if tail_words else current
            result.append(merged)
        return result

    def _overlap_tail(self, words: list[str]) -> list[str]:
        """Retourne le plus grand suffixe de `words` respectant `overlap_tokens`."""
        tail: list[str] = []
        for word in reversed(words):
            candidate = [word, *tail]
            if self._token_counter.count(" ".join(candidate)) > self._overlap_tokens:
                break
            tail = candidate
        return tail
