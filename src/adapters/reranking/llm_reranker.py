"""LLM-based reranker using Bedrock API for ablation study.

Uses LLM to evaluate chunk relevance (0-10 scale) vs cross-encoder similarity.
Slower but better semantic understanding and score separation.
"""

import os
from typing import Any

from src.domain.models import ScoredChunk


def create_llm_reranker(config: dict[str, Any]) -> "LLMReranker":
    """Factory to instantiate LLMReranker from config."""
    model_id = config.get("model_id", "global.anthropic.claude-sonnet-4-6")
    region = config.get("region", os.environ.get("AWS_REGION", "eu-west-3"))

    if "AWS_BEARER_TOKEN_BEDROCK" not in os.environ:
        raise ValueError("LLMReranker requires AWS_BEARER_TOKEN_BEDROCK in .env")

    return LLMReranker(model_id=model_id, region=region)


class LLMReranker:
    """LLM-based reranker using Bedrock Converse API.

    Evaluates each chunk relevance (0-10 scale), normalizes to [0, 1].
    """

    _RERANKING_PROMPT_TEMPLATE = """Tu es un système de classement de pertinence. Evalue la pertinence du CHUNK ci-dessous pour répondre à la QUESTION.

QUESTION : {question}

CHUNK :
{chunk_text}

Consigne : Donne un score de pertinence de 0.0 à 10.0 avec UNE décimale de précision :
- 9.0-10.0 = essentiellement pertinent, répond directement à la question
- 7.0-8.9 = très pertinent, contient des informations utiles
- 4.0-6.9 = modérément pertinent, contexte tangentiel
- 1.0-3.9 = faiblement pertinent, mention indirecte
- 0.0 = non pertinent, aucun rapport avec la question

Utilise des décimales pour affiner ton jugement (ex: 3.2, 7.8, etc.).

Réponds UNIQUEMENT par un nombre décimal entre 0.0 et 10.0, sans explication."""

    def __init__(self, model_id: str, region: str) -> None:
        """Initialise le reranker LLM.

        Args:
            model_id: ID du modele Bedrock (ex: us.anthropic.claude-sonnet-4-20250514-v1:0).
            region: Region AWS (ex: us-east-1).
        """
        self._model_id = model_id
        self._region = region
        self._llm = None  # Lazy loading (import lourd)

    def _load_llm(self) -> Any:
        """Charge le LLM Bedrock (lazy loading)."""
        if self._llm is None:
            from langchain_aws import ChatBedrockConverse

            self._llm = ChatBedrockConverse(
                model=self._model_id,
                region_name=self._region,
                temperature=0,  # Maximum de determinisme
            )
        return self._llm

    def _score_chunk(self, question: str, chunk_text: str) -> float:
        """Evalue la pertinence d'un chunk via LLM.

        Args:
            question: La question de l'utilisateur.
            chunk_text: Le texte du chunk a evaluer.

        Returns:
            Score normalise entre 0.0 et 1.0.
        """
        llm = self._load_llm()
        prompt = self._RERANKING_PROMPT_TEMPLATE.format(question=question, chunk_text=chunk_text)

        try:
            response = llm.invoke(prompt)
            response_text = response.content.strip()

            score_str = "".join(c for c in response_text if c.isdigit() or c == ".")
            if not score_str:
                return 0.5

            raw_score = float(score_str)
            raw_score = max(0.0, min(10.0, raw_score))
            return raw_score / 10.0

        except Exception:
            return 0.5

    def rerank(
        self, query: str, chunks: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        """Rerank chunks by LLM-evaluated relevance."""
        if not chunks:
            return []

        # Evalue chaque chunk avec le LLM
        rescored = []
        for scored_chunk in chunks:
            llm_score = self._score_chunk(query, scored_chunk.chunk.text)
            # Remplace le score du retriever par le score LLM
            rescored.append(ScoredChunk(chunk=scored_chunk.chunk, score=llm_score))

        # Trie par score LLM decroissant
        rescored.sort(key=lambda sc: sc.score, reverse=True)

        return rescored[:top_n]
