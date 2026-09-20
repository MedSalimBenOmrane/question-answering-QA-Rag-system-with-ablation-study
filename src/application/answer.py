"""Cas d'usage applicatif : repondre a une question via le pipeline RAG.

Relie la config (dict deja parse) aux adapters concrets (via leurs factories)
et au pipeline pur du domaine (`src/domain/pipeline.py`), puis journalise
chaque requete. Couche "online" au sens du CLAUDE.md : suppose le corpus deja
indexe (chunks + vectorstore persistants, voir `src/cli.py`).
"""

import time
from typing import Any

from src.adapters.embedding.factory import create_embedder
from src.adapters.guardrails.basic import create_guardrail
from src.adapters.llm.ollama import build_prompt, create_llm
from src.adapters.llm.token_counter import create_token_counter
from src.adapters.reranking.factory import create_reranker
from src.adapters.retrieval.factory import create_retriever
from src.adapters.vectorstore.chroma import create_vectorstore
from src.domain.budget import create_selector
from src.domain.models import Answer, Chunk
from src.domain.pipeline import Pipeline, PipelineConfig
from src.observability.logging import RequestLogger


def build_pipeline(config: dict[str, Any], chunks: list[Chunk], system_prompt: str) -> Pipeline:
    """Construit le `Pipeline` complet a partir de la config et du corpus indexe.

    Instancie chaque adapter via sa factory (aucune valeur cablee en dur) et
    les injecte dans le pipeline pur du domaine.

    Args:
        config: Configuration complete (sections embedding, vectorstore,
            retrieval, reranking, selection, token_counter, llm, guardrails,
            pipeline, budget).
        chunks: Chunks du corpus deja indexes (requis par un retriever
            lexical BM25, seul ou en hybride, pour reconstruire son index).
        system_prompt: Contenu du system prompt, deja charge par l'appelant
            (ex: `src/prompts/system.txt`).

    Returns:
        Un `Pipeline` pret a repondre aux questions.
    """
    token_counter = create_token_counter(config["token_counter"])
    embedder = create_embedder(config["embedding"])
    vectorstore = create_vectorstore(config["vectorstore"])
    retriever = create_retriever(config["retrieval"], embedder, vectorstore, chunks)
    reranker = create_reranker(config["reranking"])
    selector = create_selector(config["selection"])
    llm = create_llm(
        config["llm"],
        system_prompt=system_prompt,
        max_output_tokens=config["budget"]["reserve_answer"],
    )
    guardrail_input = create_guardrail(config["guardrails"]["input"])
    guardrail_output = create_guardrail(config["guardrails"]["output"])

    pipeline_config = PipelineConfig(
        retrieve_k=config["pipeline"]["retrieve_k"],
        rerank_top_n=config["pipeline"]["rerank_top_n"],
        budget_total=config["budget"]["total"],
        reserve_answer=config["budget"]["reserve_answer"],
    )

    return Pipeline(
        guardrail_input=guardrail_input,
        retriever=retriever,
        reranker=reranker,
        selector=selector,
        llm=llm,
        guardrail_output=guardrail_output,
        token_counter=token_counter,
        assemble_prompt=build_prompt,
        system_prompt=system_prompt,
        config=pipeline_config,
    )


def answer_question(question: str, pipeline: Pipeline, logger: RequestLogger) -> Answer:
    """Repond a une question via le pipeline, en journalisant la requete.

    Args:
        question: La question de l'utilisateur.
        pipeline: Le pipeline deja construit (voir `build_pipeline`).
        logger: Le journal structure ou consigner la requete.

    Returns:
        La reponse produite (`Answer`).
    """
    start = time.perf_counter()
    answer = pipeline.run(question)
    latency = time.perf_counter() - start

    logger.log_request(question=question, answer=answer, latency_seconds=latency)
    return answer
