"""Logs structures JSON, une ligne par requete (regle CLAUDE.md).

Chaque requete est journalisee avec : question, sources retrouvees, chunks
retenus, tokens utilises, latence, reponse, meta (scores/diagnostics).
"""

import json
import logging

from src.domain.models import Answer

_logger = logging.getLogger("context_aware_qa.requests")


class RequestLogger:
    """Journalise chaque requete en une ligne JSON structuree."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialise le logger.

        Args:
            logger: Logger Python a utiliser (par defaut : un logger dedie
                "context_aware_qa.requests", configurable via `logging.basicConfig`
                par l'appelant).
        """
        self._logger = logger or _logger

    def log_request(self, question: str, answer: Answer, latency_seconds: float) -> None:
        """Journalise une requete complete en JSON.

        Args:
            question: La question posee par l'utilisateur.
            answer: La reponse produite par le pipeline.
            latency_seconds: Duree totale de traitement de la requete.
        """
        event = {
            "question": question,
            "answer": answer.text,
            "sources": answer.sources,
            "selected_chunk_ids": answer.selected_chunk_ids,
            "tokens_used": answer.tokens_used,
            "abstained": answer.abstained,
            "latency_seconds": round(latency_seconds, 4),
            "meta": answer.meta,
        }
        self._logger.info(json.dumps(event, ensure_ascii=False))
