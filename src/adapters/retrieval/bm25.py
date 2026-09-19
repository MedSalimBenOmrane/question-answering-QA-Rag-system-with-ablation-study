"""Adapter Retriever lexical : BM25 (rank_bm25) sur l'ensemble des chunks.

Index construit en memoire a partir des chunks fournis (pas de VectorStore :
BM25 est un classement lexical, pas une recherche par similarite vectorielle).
"""

import re

from rank_bm25 import BM25Okapi

from src.domain.models import Chunk, ScoredChunk

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Tokenisation lexicale simple (mots en minuscules) pour l'indexation BM25."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Retriever lexical s'appuyant sur BM25 (rank_bm25) sur un corpus de chunks."""

    def __init__(self, chunks: list[Chunk], k1: float, b: float) -> None:
        """Construit l'index BM25 sur les chunks fournis.

        Args:
            chunks: Chunks a indexer lexicalement (deja produits par un Chunker).
            k1: Parametre de saturation de terme de BM25.
            b: Parametre de normalisation par longueur de document de BM25.
        """
        self._chunks = chunks
        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b) if tokenized_corpus else None

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        """Retourne les k chunks les plus pertinents lexicalement pour `query`.

        Args:
            query: La question de l'utilisateur.
            k: Nombre de chunks a recuperer.

        Returns:
            La liste des k chunks les plus pertinents, avec leur score BM25.
        """
        if self._bm25 is None or k <= 0:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [ScoredChunk(chunk=chunk, score=float(score)) for chunk, score in ranked[:k]]
