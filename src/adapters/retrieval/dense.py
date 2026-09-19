"""Adapter Retriever dense : recherche semantique via Embedder + VectorStore.

N'implemente aucune logique de recherche elle-meme : delegue integralement
au vecteur de requete (Embedder) et a la recherche par similarite
(VectorStore), deux ports deja implementes.
"""

from src.domain.models import ScoredChunk
from src.domain.ports import Embedder, VectorStore


class DenseRetriever:
    """Retriever dense : encode la requete puis interroge le VectorStore."""

    def __init__(self, embedder: Embedder, vectorstore: VectorStore) -> None:
        """Initialise le retriever.

        Args:
            embedder: L'Embedder utilise pour vectoriser la requete.
            vectorstore: Le VectorStore interroge pour la recherche par similarite.
        """
        self._embedder = embedder
        self._vectorstore = vectorstore

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        """Retourne les k chunks les plus proches semantiquement de `query`.

        Args:
            query: La question de l'utilisateur.
            k: Nombre de chunks a recuperer.

        Returns:
            La liste des k chunks les plus pertinents, avec leur score.
        """
        query_vector = self._embedder.embed_query(query)
        return self._vectorstore.search(query_vector, k)
