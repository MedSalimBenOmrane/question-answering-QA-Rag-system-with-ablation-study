"""Adapter Embedder base sur sentence-transformers, cible multilingual-e5.

Le modele (`sentence_transformers.SentenceTransformer`) est charge par la
factory puis injecte ici. Contrairement a BGE-M3, la famille E5 exige un
prefixe d'instruction different pour une requete ("query: ") et pour un
passage a indexer ("passage: ") : c'est ce qui distingue cet adapter de
`BgeM3Embedder` au-dela du simple nom de modele.
"""

from sentence_transformers import SentenceTransformer

_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "


class MultilingualE5Embedder:
    """Embedder s'appuyant sur un modele sentence-transformers de la famille E5."""

    def __init__(self, model: SentenceTransformer, batch_size: int) -> None:
        """Initialise l'embedder.

        Args:
            model: Modele `SentenceTransformer` deja charge (multilingual-e5-base
                par defaut).
            batch_size: Taille de batch utilisee pour l'encodage.
        """
        self._model = model
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Calcule les embeddings d'une liste de passages (indexation offline).

        Args:
            texts: Liste de textes de documents a vectoriser.

        Returns:
            La liste des vecteurs d'embedding, dans le meme ordre que `texts`.
        """
        if not texts:
            return []
        prefixed = [f"{_PASSAGE_PREFIX}{t}" for t in texts]
        vectors = self._model.encode(prefixed, batch_size=self._batch_size, convert_to_numpy=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Calcule l'embedding d'une requete utilisateur (usage online).

        Args:
            text: Le texte de la requete.

        Returns:
            Le vecteur d'embedding de la requete.
        """
        vectors = self._model.encode(
            [f"{_QUERY_PREFIX}{text}"], batch_size=self._batch_size, convert_to_numpy=True
        )
        return vectors[0].tolist()
