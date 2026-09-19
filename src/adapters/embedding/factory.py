"""Factory qui instancie la strategie d'embedding depuis la config.

Charge le vrai modele `sentence-transformers` correspondant au provider
configure (poids reels telecharges depuis le Hub HuggingFace), puis
l'enveloppe dans l'adapter `Embedder` associe. Aucune valeur n'est cablee
en dur : modele, device et taille de batch viennent tous de la config.
"""

from typing import Any

from sentence_transformers import SentenceTransformer

from src.adapters.embedding.bge_m3 import BgeM3Embedder
from src.adapters.embedding.multilingual_e5 import MultilingualE5Embedder
from src.domain.ports import Embedder

_STRATEGIES = ("bge_m3", "multilingual_e5")


def create_embedder(config: dict[str, Any]) -> Embedder:
    """Instancie l'Embedder correspondant au provider configure.

    Args:
        config: Section de configuration `embedding`. Doit contenir une cle
            `provider` parmi {"bge_m3", "multilingual_e5"}, et une sous-cle
            portant le nom du provider avec `model_name` (et optionnellement
            `device`, `batch_size`).

    Returns:
        Une instance de `Embedder` prete a l'emploi, adossee au vrai modele
        `sentence-transformers` correspondant.

    Raises:
        KeyError: Si `config` ne contient pas `strategy`/`provider`, ou si
            `model_name` est absent de la sous-config du provider choisi.
        ValueError: Si `provider` ne correspond a aucun provider connu.
    """
    try:
        provider = config["provider"]
    except KeyError as exc:
        raise KeyError("config d'embedding invalide : cle 'provider' manquante") from exc

    if provider not in _STRATEGIES:
        raise ValueError(
            f"provider d'embedding inconnu: {provider!r} (attendu parmi {_STRATEGIES})"
        )

    params = config.get(provider, {})
    try:
        model_name = params["model_name"]
    except KeyError as exc:
        raise KeyError(
            f"config d'embedding '{provider}' invalide : 'model_name' est requis"
        ) from exc

    model = SentenceTransformer(model_name, device=params.get("device"))
    batch_size = params.get("batch_size", 32)

    if provider == "bge_m3":
        return BgeM3Embedder(model=model, batch_size=batch_size)
    return MultilingualE5Embedder(model=model, batch_size=batch_size)
