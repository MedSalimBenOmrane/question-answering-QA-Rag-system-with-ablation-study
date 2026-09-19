"""Adapter Reranker base sur un cross-encoder (sentence-transformers).

Contrairement a un Embedder (qui encode requete et document separement puis
compare des vecteurs), un cross-encoder encode la paire (requete, document)
ensemble : plus couteux, mais plus precis pour reordonner un petit nombre de
candidats deja retrouves par le retrieval.
"""

from sentence_transformers import CrossEncoder

from src.domain.models import ScoredChunk


class CrossEncoderReranker:
    """Reranker s'appuyant sur un modele cross-encoder deja charge."""

    def __init__(self, model: CrossEncoder) -> None:
        """Initialise le reranker.

        Args:
            model: Modele `CrossEncoder` deja charge (ex: via
                `CrossEncoder(model_name)`).
        """
        self._model = model

    def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        """Reordonne les chunks candidats et retourne les `top_n` meilleurs.

        Args:
            query: La question de l'utilisateur.
            chunks: Liste des chunks candidats a reordonner (issus du retrieval).
            top_n: Nombre de chunks a conserver apres reranking.

        Returns:
            La liste des `top_n` chunks les plus pertinents selon le
            cross-encoder, avec son score (remplace le score du retrieval).
        """
        if not chunks or top_n <= 0:
            return []

        pairs = [(query, scored.chunk.text) for scored in chunks]
        scores = self._model.predict(pairs)

        reranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [
            ScoredChunk(chunk=scored.chunk, score=float(score))
            for scored, score in reranked[:top_n]
        ]
