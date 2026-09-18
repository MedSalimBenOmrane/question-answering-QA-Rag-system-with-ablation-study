"""Ports du coeur metier (domain).

Ce module definit les CONTRATS (interfaces) sous forme de `typing.Protocol`
que doivent respecter les adapters. Aucune implementation concrete ici.

Ne pas modifier ces interfaces sans validation explicite (cf. CLAUDE.md).
"""

from typing import Protocol

from src.domain.models import Chunk, ScoredChunk


class TokenCounter(Protocol):
    """Compte le nombre de tokens d'un texte selon le tokenizer du LLM cible."""

    def count(self, text: str) -> int:
        """Retourne le nombre de tokens contenus dans `text`.

        Args:
            text: Le texte a tokenizer.

        Returns:
            Le nombre de tokens.
        """
        ...


class Chunker(Protocol):
    """Decoupe des documents source en chunks indexables."""

    def chunk(self, docs: list[tuple[str, str]]) -> list[Chunk]:
        """Decoupe une liste de documents en chunks.

        Args:
            docs: Liste de tuples (source, texte) representant les documents.

        Returns:
            La liste des chunks produits.
        """
        ...


class Embedder(Protocol):
    """Calcule des representations vectorielles (embeddings) de textes."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Calcule les embeddings d'une liste de textes (usage offline/indexation).

        Args:
            texts: Liste de textes a vectoriser.

        Returns:
            La liste des vecteurs d'embedding, dans le meme ordre que `texts`.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Calcule l'embedding d'une requete utilisateur (usage online).

        Args:
            text: Le texte de la requete.

        Returns:
            Le vecteur d'embedding de la requete.
        """
        ...


class VectorStore(Protocol):
    """Stocke et interroge des vecteurs associes a des chunks."""

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Ajoute des chunks et leurs vecteurs associes au store.

        Args:
            chunks: Liste des chunks a indexer.
            vectors: Liste des vecteurs d'embedding, alignee avec `chunks`.
        """
        ...

    def search(self, query_vector: list[float], k: int) -> list[ScoredChunk]:
        """Recherche les k chunks les plus proches d'un vecteur de requete.

        Args:
            query_vector: Vecteur d'embedding de la requete.
            k: Nombre de resultats a retourner.

        Returns:
            La liste des k chunks les plus pertinents, avec leur score.
        """
        ...


class Retriever(Protocol):
    """Recupere les chunks pertinents pour une question donnee."""

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        """Recupere les k chunks les plus pertinents pour une requete.

        Args:
            query: La question de l'utilisateur.
            k: Nombre de chunks a recuperer.

        Returns:
            La liste des chunks retrouves, avec leur score.
        """
        ...


class Reranker(Protocol):
    """Reordonne une liste de chunks candidats selon leur pertinence a la requete."""

    def rerank(
        self, query: str, chunks: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        """Reordonne les chunks candidats et retourne les `top_n` meilleurs.

        Args:
            query: La question de l'utilisateur.
            chunks: Liste des chunks candidats a reordonner.
            top_n: Nombre de chunks a conserver apres reranking.

        Returns:
            La liste des `top_n` chunks les plus pertinents, reordonnes.
        """
        ...


class Selector(Protocol):
    """Selectionne les chunks a inclure dans le contexte, sous budget de tokens."""

    def select(
        self, query: str, chunks: list[ScoredChunk], context_budget: int
    ) -> list[Chunk]:
        """Selectionne un sous-ensemble de chunks respectant le budget de tokens.

        Args:
            query: La question de l'utilisateur.
            chunks: Liste des chunks candidats, avec leur score.
            context_budget: Budget de tokens disponible pour le contexte.

        Returns:
            La liste des chunks selectionnes pour construire le contexte.
        """
        ...


class LLM(Protocol):
    """Genere du texte a partir d'un prompt (modele local via Ollama)."""

    def generate(self, prompt: str) -> str:
        """Genere une reponse textuelle a partir d'un prompt.

        Args:
            prompt: Le prompt complet (question + contexte + instructions).

        Returns:
            Le texte genere par le LLM.
        """
        ...


class Guardrail(Protocol):
    """Verifie qu'un texte (question ou reponse) respecte des regles de securite/qualite."""

    def check(self, text: str) -> tuple[bool, str]:
        """Verifie la validite d'un texte.

        Args:
            text: Le texte a verifier.

        Returns:
            Un tuple (ok, raison) : `ok` est True si le texte est valide,
            `raison` explique le motif de rejet (chaine vide si `ok` est True).
        """
        ...
