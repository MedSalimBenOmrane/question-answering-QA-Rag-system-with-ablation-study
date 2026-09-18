"""Modeles de donnees du coeur metier (domain).

Ce module ne depend d'aucune bibliotheque externe : uniquement des dataclasses
standard. Il est le contrat de donnees partage par les ports et les adapters.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """Un fragment de document indexable, issu d'une strategie de chunking.

    Attributes:
        id: Identifiant unique du chunk.
        text: Contenu textuel du chunk.
        source: Chemin ou identifiant du document d'origine.
        n_tokens: Nombre de tokens du texte, mesure avec le tokenizer du LLM cible.
        metadata: Metadonnees additionnelles libres (ex: position, titre de section).
    """

    id: str
    text: str
    source: str
    n_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredChunk:
    """Un chunk associe a un score de pertinence (retrieval, reranking, fusion).

    Attributes:
        chunk: Le chunk concerne.
        score: Score de pertinence associe (plus haut = plus pertinent).
    """

    chunk: Chunk
    score: float


@dataclass
class Answer:
    """La reponse produite par le pipeline pour une question donnee.

    Attributes:
        text: Texte de la reponse generee (ou message d'abstention).
        sources: Liste des sources (documents) citees dans la reponse.
        selected_chunk_ids: Identifiants des chunks retenus dans le contexte final.
        tokens_used: Nombre de tokens consommes pour produire la reponse.
        abstained: True si le systeme s'est abstenu de repondre.
        meta: Metadonnees additionnelles (scores, latence, config utilisee, etc.).
    """

    text: str
    sources: list[str]
    selected_chunk_ids: list[str]
    tokens_used: int
    abstained: bool
    meta: dict[str, Any] = field(default_factory=dict)
