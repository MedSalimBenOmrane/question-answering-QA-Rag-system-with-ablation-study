"""TokenCounter concret, base sur le tokenizer du modele LLM cible.

Deux implementations :
- `TiktokenTokenCounter` : encodage tiktoken (modeles compatibles OpenAI).
- `HFTokenCounter` : tokenizer HuggingFace (modeles locaux servis par Ollama).

Une factory (`create_token_counter`) choisit l'implementation depuis la config,
jamais recodee ici : on delegue toujours au tokenizer reel du modele.
"""

from typing import Any, Protocol


class _Encoder(Protocol):
    """Contrat minimal partage par un encodage tiktoken et un tokenizer HF."""

    def encode(self, text: str) -> list[int]:
        ...


class TiktokenTokenCounter:
    """Compte les tokens via un encodage tiktoken (modeles compatibles OpenAI)."""

    def __init__(self, encoding: _Encoder) -> None:
        """Initialise le compteur.

        Args:
            encoding: Encodage tiktoken deja charge (ex: via `tiktoken.get_encoding`).
        """
        self._encoding = encoding

    def count(self, text: str) -> int:
        """Retourne le nombre de tokens de `text` selon l'encodage tiktoken.

        Args:
            text: Le texte a tokenizer.

        Returns:
            Le nombre de tokens.
        """
        return len(self._encoding.encode(text))


class HFTokenCounter:
    """Compte les tokens via un tokenizer HuggingFace (modeles locaux/Ollama)."""

    def __init__(self, tokenizer: _Encoder) -> None:
        """Initialise le compteur.

        Args:
            tokenizer: Tokenizer HuggingFace deja charge (ex: via
                `transformers.AutoTokenizer.from_pretrained`), correspondant
                au modele reellement servi par Ollama.
        """
        self._tokenizer = tokenizer

    def count(self, text: str) -> int:
        """Retourne le nombre de tokens de `text` selon le tokenizer HF.

        Args:
            text: Le texte a tokenizer.

        Returns:
            Le nombre de tokens.
        """
        return len(self._tokenizer.encode(text))


def create_token_counter(config: dict[str, Any]) -> TiktokenTokenCounter | HFTokenCounter:
    """Instancie le `TokenCounter` correspondant au provider configure.

    Args:
        config: Section de configuration `token_counter`. Doit contenir une cle
            `provider` parmi {"tiktoken", "hf"}, et la sous-config associee :
            - `tiktoken`: {"encoding": "<nom d'encodage tiktoken>"}
            - `hf`: {"model_name": "<nom ou chemin du modele HF>"}

    Returns:
        Une instance de `TokenCounter` prete a l'emploi.

    Raises:
        KeyError: Si une cle de configuration requise est manquante.
        ValueError: Si `provider` ne correspond a aucun provider connu.
    """
    try:
        provider = config["provider"]
    except KeyError as exc:
        raise KeyError("config de token_counter invalide : cle 'provider' manquante") from exc

    if provider == "tiktoken":
        import tiktoken

        try:
            encoding_name = config["tiktoken"]["encoding"]
        except KeyError as exc:
            raise KeyError(
                "config de token_counter 'tiktoken' invalide : 'encoding' est requis"
            ) from exc
        return TiktokenTokenCounter(tiktoken.get_encoding(encoding_name))

    if provider == "hf":
        from transformers import AutoTokenizer

        try:
            model_name = config["hf"]["model_name"]
        except KeyError as exc:
            raise KeyError(
                "config de token_counter 'hf' invalide : 'model_name' est requis"
            ) from exc
        return HFTokenCounter(AutoTokenizer.from_pretrained(model_name))

    raise ValueError(
        f"provider de token_counter inconnu: {provider!r} (attendu 'tiktoken' ou 'hf')"
    )
