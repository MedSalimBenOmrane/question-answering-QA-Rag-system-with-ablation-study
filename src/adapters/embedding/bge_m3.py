"""Adapter Embedder base sur sentence-transformers, cible BGE-M3.

Le nom du modele (ex: "BAAI/bge-m3"), le device et la taille de batch
viennent tous de la config : rien n'est cable en dur.
"""

from typing import Any, Protocol


class _SentenceEncoder(Protocol):
    """Contrat minimal attendu d'un modele sentence-transformers charge."""

    def encode(self, texts: list[str], batch_size: int, convert_to_numpy: bool) -> Any:
        ...


class BgeM3Embedder:
    """Embedder s'appuyant sur un modele sentence-transformers (BGE-M3 par defaut)."""

    def __init__(self, model: _SentenceEncoder, batch_size: int) -> None:
        """Initialise l'embedder.

        Args:
            model: Modele sentence-transformers deja charge (ex: via
                `SentenceTransformer(model_name)`).
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


def create_embedder(config: dict[str, Any]) -> BgeM3Embedder:
    """Instancie l'Embedder configure (modele sentence-transformers).

    Args:
        config: Section de configuration `embedding`. Doit contenir
            `model_name` (ex: "BAAI/bge-m3") ; `device` et `batch_size` sont
            optionnels.

    Returns:
        Un `BgeM3Embedder` pret a l'emploi.

    Raises:
        KeyError: Si `model_name` est absent de la config.
    """
    from sentence_transformers import SentenceTransformer

    try:
        model_name = config["model_name"]
    except KeyError as exc:
        raise KeyError("config d'embedding invalide : 'model_name' est requis") from exc

    model = SentenceTransformer(model_name, device=config.get("device"))
    return BgeM3Embedder(model=model, batch_size=config.get("batch_size", 32))
