"""Adapter Guardrail minimal : verifications de bon sens (texte non vide, longueur).

Aucune regle de securite/qualite avancee n'a ete specifiee pour ce projet
(detection de PII, injection de prompt, moderation de contenu...) : ce
guardrail est un placeholder volontairement simple, suffisant pour cabler le
pipeline (guardrail_input / guardrail_output), a etendre si des regles plus
riches sont demandees.
"""

from typing import Any


class BasicGuardrail:
    """Rejette un texte vide/blanc ou depassant une longueur maximale."""

    def __init__(self, max_length: int) -> None:
        """Initialise le guardrail.

        Args:
            max_length: Longueur maximale (en caracteres) du texte accepte.
        """
        self._max_length = max_length

    def check(self, text: str) -> tuple[bool, str]:
        """Verifie que `text` n'est ni vide ni trop long.

        Args:
            text: Le texte a verifier (question ou reponse).

        Returns:
            Un tuple (ok, raison). `raison` est vide si `ok` est True.
        """
        if not text or not text.strip():
            return False, "Le texte est vide."
        if len(text) > self._max_length:
            return False, (
                f"Le texte depasse la longueur maximale autorisee "
                f"({self._max_length} caracteres)."
            )
        return True, ""


def create_guardrail(config: dict[str, Any]) -> BasicGuardrail:
    """Instancie le Guardrail configure.

    Args:
        config: Section de configuration (ex: `guardrails.input` ou
            `guardrails.output`). Doit contenir `max_length`.

    Returns:
        Un `BasicGuardrail` pret a l'emploi.

    Raises:
        KeyError: Si `max_length` est absent de la config.
    """
    try:
        max_length = config["max_length"]
    except KeyError as exc:
        raise KeyError("config de guardrail invalide : 'max_length' est requis") from exc

    return BasicGuardrail(max_length=max_length)
