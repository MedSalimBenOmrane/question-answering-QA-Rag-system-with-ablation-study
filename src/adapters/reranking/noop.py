"""Adapter Reranker neutre : reranking desactive.

Utilise quand `reranking.enabled` vaut faux en config, pour que le pipeline
puisse toujours appeler un `Reranker` sans branche conditionnelle propre a
cette etape (le choix se fait entierement a la Factory).
"""

from src.domain.models import ScoredChunk


class NoOpReranker:
    """Reranker qui ne reordonne rien : conserve l'ordre du retrieval brut."""

    def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        """Retourne les `top_n` premiers chunks, sans reordonnancement.

        Args:
            query: La question de l'utilisateur (non utilisee, reranking desactive).
            chunks: Liste des chunks candidats, deja ordonnes par le retrieval.
            top_n: Nombre de chunks a conserver.

        Returns:
            Les `top_n` premiers `chunks`, dans leur ordre d'origine.
        """
        return chunks[:top_n]
