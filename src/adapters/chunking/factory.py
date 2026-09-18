"""Factory qui instancie la strategie de chunking depuis la config.

Aucune valeur n'est cablee en dur : tout (strategie, taille de chunk,
overlap) provient du dictionnaire de config passe en argument.
"""

from typing import Any

from src.domain.ports import Chunker, TokenCounter
from src.adapters.chunking.fixed import FixedSizeChunker
from src.adapters.chunking.markdown import MarkdownChunker
from src.adapters.chunking.recursive import RecursiveChunker

_STRATEGIES = ("fixed", "recursive", "markdown")


def create_chunker(config: dict[str, Any], token_counter: TokenCounter) -> Chunker:
    """Instancie l'adapter `Chunker` correspondant a la strategie configuree.

    Args:
        config: Section de configuration `chunking` (ex: issue de `default.yaml`).
            Doit contenir une cle `strategy` parmi {"fixed", "recursive", "markdown"},
            et le cas echeant une sous-cle portant le nom de la strategie avec
            ses parametres propres (ex: `chunk_size_tokens`, `overlap_tokens`).
        token_counter: Le `TokenCounter` a injecter dans l'adapter cree.

    Returns:
        Une instance de `Chunker` prete a l'emploi.

    Raises:
        KeyError: Si `config` ne contient pas la cle `strategy`, ou si les
            parametres requis par la strategie choisie sont absents.
        ValueError: Si `strategy` ne correspond a aucune strategie connue.
    """
    try:
        strategy = config["strategy"]
    except KeyError as exc:
        raise KeyError(
            "config de chunking invalide : cle 'strategy' manquante"
        ) from exc

    if strategy not in _STRATEGIES:
        raise ValueError(
            f"strategie de chunking inconnue: {strategy!r} (attendu parmi {_STRATEGIES})"
        )

    if strategy == "fixed":
        params = config.get("fixed", {})
        try:
            return FixedSizeChunker(
                token_counter=token_counter,
                chunk_size_tokens=params["chunk_size_tokens"],
                overlap_tokens=params["overlap_tokens"],
            )
        except KeyError as exc:
            raise KeyError(
                "config de chunking 'fixed' invalide : "
                "'chunk_size_tokens' et 'overlap_tokens' sont requis"
            ) from exc

    if strategy == "recursive":
        params = config.get("recursive", {})
        try:
            chunk_size_tokens = params["chunk_size_tokens"]
        except KeyError as exc:
            raise KeyError(
                "config de chunking 'recursive' invalide : 'chunk_size_tokens' est requis"
            ) from exc
        return RecursiveChunker(
            token_counter=token_counter,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=params.get("overlap_tokens", 0),
        )

    return MarkdownChunker(token_counter=token_counter)
