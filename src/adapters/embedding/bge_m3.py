"""Adapter Embedder base sur sentence-transformers, cible BGE-M3.

Le modele (`sentence_transformers.SentenceTransformer`) est charge par la
factory (`src/adapters/embedding/factory.py`) puis injecte ici : cette
classe ne fait qu'appliquer le contrat `Embedder` par-dessus le modele reel,
elle ne redefinit jamais son propre encodage.
"""

from sentence_transformers import SentenceTransformer


class BgeM3Embedder:
    """Embedder s'appuyant sur un modele sentence-transformers (BGE-M3 par defaut)."""

    def __init__(self, model: SentenceTransformer, batch_size: int) -> None:
        """Initialise l'embedder.

        Args:
            model: Modele `SentenceTransformer` deja charge (BGE-M3 par defaut).
            batch_size: Taille de batch utilisee pour l'encodage.
        """
        self._model = model
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Calcule les embeddings d'une liste de textes (indexation offline).

        Args:
            texts: Liste de textes a vectoriser.

        Returns:
            La liste des vecteurs d'embedding, dans le meme ordre que `texts`.
        """
        if not texts:
            return []
        vectors = self._model.encode(texts, batch_size=self._batch_size, convert_to_numpy=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Calcule l'embedding d'une requete utilisateur (usage online).

        Args:
            text: Le texte de la requete.

        Returns:
            Le vecteur d'embedding de la requete.
        """
        vectors = self._model.encode([text], batch_size=self._batch_size, convert_to_numpy=True)
        return vectors[0].tolist()
