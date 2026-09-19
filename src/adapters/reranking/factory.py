"""Factory qui instancie le Reranker depuis la config (active ou non).

Aucune valeur n'est cablee en dur : l'activation, le nom du modele et le
device viennent tous de la config.
"""

from typing import Any

from sentence_transformers import CrossEncoder

from src.adapters.reranking.cross_encoder import CrossEncoderReranker
from src.adapters.reranking.noop import NoOpReranker
from src.domain.ports import Reranker


def create_reranker(config: dict[str, Any]) -> Reranker:
    """Instancie le Reranker configure (cross-encoder reel, ou neutre si desactive).

    Args:
        config: Section de configuration `reranking`. Doit contenir `enabled`
            (bool). Si vrai, doit aussi contenir `model_name` (et
            optionnellement `device`).

    Returns:
        Un `CrossEncoderReranker` si `enabled` est vrai, sinon un `NoOpReranker`.

    Raises:
        KeyError: Si `enabled` est absent, ou si `model_name` est absent alors
            que `enabled` est vrai.
    """
    try:
        enabled = config["enabled"]
    except KeyError as exc:
        raise KeyError("config de reranking invalide : cle 'enabled' manquante") from exc

    if not enabled:
        return NoOpReranker()

    try:
        model_name = config["model_name"]
    except KeyError as exc:
        raise KeyError(
            "config de reranking invalide : 'model_name' est requis si enabled=true"
        ) from exc

    model = CrossEncoder(model_name, device=config.get("device"))
    return CrossEncoderReranker(model=model)
