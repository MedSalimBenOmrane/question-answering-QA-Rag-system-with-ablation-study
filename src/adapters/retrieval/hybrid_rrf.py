"""Adapter Retriever hybride : fusion de deux classements par Reciprocal Rank Fusion.

RRF combine les classements de deux retrievers (typiquement dense + lexical)
sans avoir besoin de comparer leurs scores bruts (echelles differentes) :
seul le RANG de chaque chunk dans chaque classement compte.
"""

from src.domain.models import Chunk, ScoredChunk
from src.domain.ports import Retriever


class HybridRRFRetriever:
    """Retriever hybride : fusionne un retriever dense et un retriever lexical par RRF."""

    def __init__(self, dense: Retriever, sparse: Retriever, rrf_k: int) -> None:
        """Initialise le retriever hybride.

        Args:
            dense: Retriever dense (recherche semantique).
            sparse: Retriever lexical (ex: BM25).
            rrf_k: Constante de lissage de la formule RRF : score += 1 / (rrf_k + rang).
                Plus elle est grande, plus l'ecart entre les premiers rangs est attenue.
        """
        self._dense = dense
        self._sparse = sparse
        self._rrf_k = rrf_k

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        """Fusionne les classements dense et lexical par Reciprocal Rank Fusion.

        Args:
            query: La question de l'utilisateur.
            k: Nombre de chunks a recuperer dans le classement fusionne.

        Returns:
            La liste des k chunks les plus pertinents apres fusion, avec leur
            score RRF (somme de 1 / (rrf_k + rang) sur les deux classements).
        """
        dense_results = self._dense.retrieve(query, k)
        sparse_results = self._sparse.retrieve(query, k)

        fused_scores: dict[str, float] = {}
        chunk_by_id: dict[str, Chunk] = {}

        for results in (dense_results, sparse_results):
            for rank, scored in enumerate(results, start=1):
                chunk_id = scored.chunk.id
                # Si un chunk est retrouve par les deux retrievers, garder de
                # preference l'exemplaire qui porte un embedding (cote dense) :
                # le sparse (ex: BM25) n'en porte pas, et l'ecraser ferait
                # perdre metadata["embedding"], requis par la strategie de
                # selection knapsack_mmr (cf. src/domain/budget.py).
                existing = chunk_by_id.get(chunk_id)
                if existing is None or (
                    "embedding" not in existing.metadata and "embedding" in scored.chunk.metadata
                ):
                    chunk_by_id[chunk_id] = scored.chunk
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (
                    self._rrf_k + rank
                )

        ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
        return [
            ScoredChunk(chunk=chunk_by_id[cid], score=fused_scores[cid])
            for cid in ranked_ids[:k]
        ]
