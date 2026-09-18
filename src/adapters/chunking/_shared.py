"""Utilitaires internes partages par les strategies de chunking.

Prive au package `adapters.chunking` : ne fait pas partie des ports.
"""

import re
from typing import Any

from src.domain.models import Chunk
from src.domain.ports import TokenCounter

_SOURCE_MARKER_RE = re.compile(r"<!--\s*SOURCE:\s*(\S+).*?-->", re.IGNORECASE)


def split_by_source_marker(default_source: str, text: str) -> list[tuple[str, str]]:
    """Decoupe un texte en sous-documents selon les marqueurs `<!-- SOURCE: id -->`.

    Le corpus peut concatener plusieurs documents dans un seul fichier, chacun
    precede d'un marqueur `<!-- SOURCE: fileNN ... -->`. Cette fonction isole
    chaque sous-document et lui associe le bon identifiant de source.

    Args:
        default_source: Source a utiliser si aucun marqueur n'est trouve
            (ex: le chemin du fichier physique).
        text: Le texte complet, pouvant contenir 0, 1 ou plusieurs marqueurs.

    Returns:
        Une liste de tuples (source, texte) ; le texte de chaque segment ne
        contient plus la ligne de marqueur.
    """
    matches = list(_SOURCE_MARKER_RE.finditer(text))
    if not matches:
        return [(default_source, text)] if text.strip() else []

    segments: list[tuple[str, str]] = []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        segments.append((default_source, preamble))

    for idx, match in enumerate(matches):
        segment_start = match.end()
        segment_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        segment_text = text[segment_start:segment_end].strip()
        if segment_text:
            segments.append((match.group(1), segment_text))

    return segments


def make_chunk(
    source: str,
    text: str,
    index: int,
    token_counter: TokenCounter,
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    """Construit un `Chunk` avec un id deterministe et son nombre de tokens.

    Args:
        source: Source du chunk (id de document).
        text: Contenu textuel du chunk.
        index: Position du chunk au sein de sa source (pour l'id).
        token_counter: Le `TokenCounter` injecte, utilise pour compter les tokens.
        metadata: Metadonnees additionnelles a attacher au chunk.

    Returns:
        Le `Chunk` construit.
    """
    return Chunk(
        id=f"{source}#{index}",
        text=text,
        source=source,
        n_tokens=token_counter.count(text),
        metadata=metadata or {},
    )
