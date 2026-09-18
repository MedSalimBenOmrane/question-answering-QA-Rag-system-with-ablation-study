"""Strategie de chunking a taille fixe (en tokens) avec overlap.

Adapter du port `Chunker`. Baseline naive : decoupe le texte en fenetres de
mots dont le nombre de tokens (mesure via le `TokenCounter` injecte) respecte
`chunk_size_tokens`, avec un chevauchement `overlap_tokens` entre fenetres
consecutives.
"""

from src.domain.models import Chunk
from src.domain.ports import TokenCounter
from src.adapters.chunking._shared import make_chunk, split_by_source_marker


class FixedSizeChunker:
    """Chunker a taille fixe (en tokens) avec chevauchement (overlap)."""

    def __init__(
        self,
        token_counter: TokenCounter,
        chunk_size_tokens: int,
        overlap_tokens: int,
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
        """Decoupe les documents en chunks de taille fixe avec overlap.

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
        words = text.split()
        if not words:
            return []

        result: list[Chunk] = []
        i = 0
        n = len(words)
        while i < n:
            j = self._grow_window(words, i, self._chunk_size_tokens)
            window_text = " ".join(words[i:j])
            result.append(make_chunk(source, window_text, len(result), self._token_counter))

            if j >= n:
                break

            next_i = self._overlap_start(words, i, j)
            i = next_i if next_i > i else i + 1

        return result

    def _grow_window(self, words: list[str], start: int, token_budget: int) -> int:
        """Retourne l'indice de fin (exclu) de la plus grande fenetre <= budget."""
        end = start
        while end < len(words):
            candidate = " ".join(words[start : end + 1])
            if self._token_counter.count(candidate) > token_budget and end > start:
                break
            end += 1
        return end

    def _overlap_start(self, words: list[str], window_start: int, window_end: int) -> int:
        """Retourne l'indice de depart de la fenetre suivante, avec overlap."""
        if self._overlap_tokens <= 0:
            return window_end

        start = window_end
        while start > window_start:
            candidate = " ".join(words[start - 1 : window_end])
            if self._token_counter.count(candidate) > self._overlap_tokens:
                break
            start -= 1
        return start
