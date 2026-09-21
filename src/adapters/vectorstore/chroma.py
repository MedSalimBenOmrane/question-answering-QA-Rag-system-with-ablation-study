"""Adapter VectorStore base sur Chroma, persistant selon la config.

Le chemin de persistance et le nom de la collection viennent de la config.
La collection est configuree en espace cosinus, pour que le score retourne
par `search` (1 - distance) soit directement une similarite (plus haut =
plus pertinent).
"""

from typing import Any

from src.domain.models import Chunk, ScoredChunk


class ChromaVectorStore:
    """VectorStore s'appuyant sur une collection Chroma persistante."""

    def __init__(self, client: Any, collection_name: str) -> None:
        """Initialise le store sur une collection Chroma (creee si absente).

        Args:
            client: Client Chroma deja instancie (ex: `chromadb.PersistentClient`).
            collection_name: Nom de la collection a utiliser.
        """
        self._client = client
        self._collection_name = collection_name
        self._collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Ajoute des chunks et leurs vecteurs a la collection.

        Args:
            chunks: Liste des chunks a indexer.
            vectors: Liste des vecteurs d'embedding, alignee avec `chunks`.

        Raises:
            ValueError: Si `chunks` et `vectors` n'ont pas la meme longueur.
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks et vectors doivent avoir la meme longueur")
        if not chunks:
            return

        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[
                {"source": c.source, "n_tokens": c.n_tokens, **c.metadata} for c in chunks
            ],
        )

    def reset(self) -> None:
        """Delete and recreate the collection (clears all data)."""
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )

    def search(self, query_vector: list[float], k: int) -> list[ScoredChunk]:
        """Recherche les k chunks les plus proches d'un vecteur de requete.

        Args:
            query_vector: Vecteur d'embedding de la requete.
            k: Nombre de resultats a retourner.

        Returns:
            La liste des k chunks les plus pertinents, avec leur score de
            similarite cosinus (plus haut = plus pertinent). Le vecteur
            d'embedding de chaque chunk est inclus dans
            `chunk.metadata["embedding"]`, disponible pour un usage eventuel
            par une strategie de selection (cf. `src/domain/budget.py`).
        """
        if k <= 0:
            return []

        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        embeddings = result["embeddings"][0]

        scored_chunks: list[ScoredChunk] = []
        for chunk_id, text, raw_metadata, distance, embedding in zip(
            ids, documents, metadatas, distances, embeddings
        ):
            metadata = dict(raw_metadata)
            source = metadata.pop("source")
            n_tokens = metadata.pop("n_tokens")
            metadata["embedding"] = list(embedding)
            chunk = Chunk(
                id=chunk_id, text=text, source=source, n_tokens=n_tokens, metadata=metadata
            )
            scored_chunks.append(ScoredChunk(chunk=chunk, score=1.0 - distance))

        return scored_chunks


def create_vectorstore(config: dict[str, Any]) -> ChromaVectorStore:
    """Instancie le VectorStore configure (collection Chroma persistante).

    La telemetrie anonyme de Chroma est desactivee (pas d'appel reseau sortant).

    Args:
        config: Section de configuration `vectorstore`. Doit contenir
            `persist_directory` et `collection_name`.

    Returns:
        Un `ChromaVectorStore` pret a l'emploi.

    Raises:
        KeyError: Si `persist_directory` ou `collection_name` sont absents.
    """
    import chromadb
    from chromadb.config import Settings

    try:
        persist_directory = config["persist_directory"]
        collection_name = config["collection_name"]
    except KeyError as exc:
        raise KeyError(
            "config de vectorstore invalide : 'persist_directory' et "
            "'collection_name' sont requis"
        ) from exc

    client = chromadb.PersistentClient(
        path=persist_directory, settings=Settings(anonymized_telemetry=False)
    )
    return ChromaVectorStore(client=client, collection_name=collection_name)
