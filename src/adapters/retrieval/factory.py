"""Factory qui instancie la strategie de retrieval depuis la config.

Aucune valeur n'est cablee en dur : le choix hybride/simple, la strategie
non-hybride (dense | bm25), les parametres BM25 (k1, b) et la constante
RRF (rrf_k) viennent tous de la config.
"""

from typing import Any

from src.adapters.retrieval.bm25 import BM25Retriever
from src.adapters.retrieval.dense import DenseRetriever
from src.adapters.retrieval.hybrid_rrf import HybridRRFRetriever
from src.domain.models import Chunk
from src.domain.ports import Embedder, Retriever, VectorStore

_STRATEGIES = ("dense", "bm25")


def create_retriever(
    config: dict[str, Any],
    embedder: Embedder,
    vectorstore: VectorStore,
    chunks: list[Chunk],
) -> Retriever:
    """Instancie le Retriever configure (dense, bm25, ou hybride RRF des deux).

    Args:
        config: Section de configuration `retrieval`. Doit contenir `hybrid`
            (bool). Si `hybrid` est vrai, doit aussi contenir `rrf_k` et
            `bm25` (k1, b). Si `hybrid` est faux, doit contenir `strategy`
            parmi {"dense", "bm25"} (et `bm25` si `strategy` vaut "bm25").
        embedder: L'Embedder a injecter dans le retriever dense.
        vectorstore: Le VectorStore a injecter dans le retriever dense.
        chunks: Les chunks a indexer lexicalement pour BM25.

    Returns:
        Une instance de `Retriever` prete a l'emploi.

    Raises:
        KeyError: Si une cle de configuration requise est manquante.
        ValueError: Si `strategy` (mode non-hybride) ne correspond a aucune
            strategie connue.
    """
    try:
        hybrid = config["hybrid"]
    except KeyError as exc:
        raise KeyError("config de retrieval invalide : cle 'hybrid' manquante") from exc

    def _build_dense() -> DenseRetriever:
        return DenseRetriever(embedder=embedder, vectorstore=vectorstore)

    def _build_bm25() -> BM25Retriever:
        try:
            bm25_params = config["bm25"]
            k1 = bm25_params["k1"]
            b = bm25_params["b"]
        except KeyError as exc:
            raise KeyError(
                "config de retrieval 'bm25' invalide : 'k1' et 'b' sont requis"
            ) from exc
        return BM25Retriever(chunks=chunks, k1=k1, b=b)

    if hybrid:
        try:
            rrf_k = config["rrf_k"]
        except KeyError as exc:
            raise KeyError(
                "config de retrieval invalide : 'rrf_k' est requis si hybrid=true"
            ) from exc
        return HybridRRFRetriever(dense=_build_dense(), sparse=_build_bm25(), rrf_k=rrf_k)

    try:
        strategy = config["strategy"]
    except KeyError as exc:
        raise KeyError(
            "config de retrieval invalide : 'strategy' est requis si hybrid=false"
        ) from exc

    if strategy == "dense":
        return _build_dense()
    if strategy == "bm25":
        return _build_bm25()
    raise ValueError(
        f"strategie de retrieval inconnue: {strategy!r} (attendu parmi {_STRATEGIES})"
    )
