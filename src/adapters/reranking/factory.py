"""Factory qui instancie le Reranker depuis la config (active ou non).

Aucune valeur n'est cablee en dur : l'activation, le provider (cross-encoder
ou llm) et les parametres viennent tous de la config.

Providers supportes :
- "cross-encoder" (defaut) : CrossEncoderReranker (sentence-transformers)
- "llm" : LLMReranker (Bedrock Claude Sonnet 4.6, pour etude d'ablation)
"""

from typing import Any

from sentence_transformers import CrossEncoder

from src.adapters.reranking.cross_encoder import CrossEncoderReranker
from src.adapters.reranking.llm_reranker import create_llm_reranker
from src.adapters.reranking.noop import NoOpReranker
from src.domain.ports import Reranker


def create_reranker(config: dict[str, Any]) -> Reranker:
    """Instancie le Reranker configure (cross-encoder, llm, ou noop si desactive).

    Args:
        config: Section de configuration `reranking`. Doit contenir `enabled`
            (bool). Si vrai, peut contenir `provider` ("cross-encoder" par
            defaut, ou "llm"). Si provider="cross-encoder", requiert
            `model_name`. Si provider="llm", requiert credentials Bedrock.

    Returns:
        Un Reranker selon la config : CrossEncoderReranker, LLMReranker, ou
        NoOpReranker si desactive.

    Raises:
        KeyError: Si `enabled` est absent, ou parametres manquants selon provider.
        ValueError: Si provider inconnu ou credentials manquants.
    """
    try:
        enabled = config["enabled"]
    except KeyError as exc:
        raise KeyError("config de reranking invalide : cle 'enabled' manquante") from exc

    if not enabled:
        return NoOpReranker()

    # Determine le provider (defaut: cross-encoder)
    provider = config.get("provider", "cross-encoder")

    if provider == "cross-encoder":
        # Reranker classique (sentence-transformers)
        try:
            model_name = config["model_name"]
        except KeyError as exc:
            raise KeyError(
                "config de reranking invalide : 'model_name' est requis "
                "si enabled=true et provider='cross-encoder'"
            ) from exc

        model = CrossEncoder(model_name, device=config.get("device"))
        return CrossEncoderReranker(model=model)

    elif provider == "llm":
        # Reranker LLM (Bedrock, pour ablation)
        return create_llm_reranker(config)

    else:
        raise ValueError(
            f"provider de reranking inconnu: {provider!r} "
            "(attendu parmi 'cross-encoder', 'llm')"
        )
