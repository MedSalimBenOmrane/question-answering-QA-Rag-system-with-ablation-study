"""Pipeline RAG lineaire (coeur du domaine) : orchestre les ports, jamais les
adapters concrets ni aucune bibliotheque externe.

Chaine : guardrail_input -> retrieve -> rerank -> (context_budget) -> select
-> assemble prompt -> llm -> guardrail_output -> Answer.

Note d'architecture : une orchestration "LangChain (LCEL)". Ces deux regles
se contredisent pour ce fichier explicitement place dans domain/. Le choix
fait ici privilegie la purete du domaine (regle sans exception dans le
: ce pipeline est une composition lineaire de pure Python (aucun
import LangChain). Rien n'empeche une future couche d'orchestration LCEL
d'envelopper `Pipeline.run` dans `src/application/

Toutes les dependances (ports + fonction d'assemblage de prompt + system
prompt deja charge) sont injectees au constructeur : ce module ne fait ni
I/O, ni instanciation d'adapter, ni lecture de fichier/config.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable

from src.domain.models import Answer, Chunk
from src.domain.ports import LLM, Guardrail, Reranker, Retriever, Selector, TokenCounter

ABSTENTION_TEXT = "Information non trouvée dans les documents."


@dataclass
class PipelineConfig:
    """Parametres numeriques du pipeline (jamais codes en dur dans `Pipeline`).

    Attributes:
        retrieve_k: Nombre de chunks a recuperer par le Retriever.
        rerank_top_n: Nombre de chunks conserves apres reranking.
        budget_total: Budget total de tokens pour la requete (prompt systeme
            + contexte + question + reserve pour la reponse).
        reserve_answer: Tokens reserves pour la generation de la reponse,
            soustraits du budget total avant de calculer `context_budget`.
    """

    retrieve_k: int
    rerank_top_n: int
    budget_total: int
    reserve_answer: int


class Pipeline:
    """Pipeline RAG lineaire : question -> Answer, sous budget de tokens strict."""

    def __init__(
        self,
        guardrail_input: Guardrail,
        retriever: Retriever,
        reranker: Reranker,
        selector: Selector,
        llm: LLM,
        guardrail_output: Guardrail,
        token_counter: TokenCounter,
        assemble_prompt: Callable[[list[Chunk], str], str],
        system_prompt: str,
        config: PipelineConfig,
    ) -> None:
        """Initialise le pipeline avec toutes ses dependances (ports injectes).

        Args:
            guardrail_input: Verifie la question avant tout traitement.
            retriever: Recupere les chunks candidats.
            reranker: Reordonne les chunks candidats.
            selector: Selectionne les chunks sous `context_budget`.
            llm: Genere la reponse a partir du prompt assemble.
            guardrail_output: Verifie la reponse generee avant de la renvoyer.
            token_counter: Compte les tokens (system prompt, question) pour
                calculer `context_budget`.
            assemble_prompt: Fonction (chunks, question) -> prompt. Injectee
                plutot qu'importee, pour ne dependre d'aucun adapter LLM
                concret (ex: `src.adapters.llm.ollama.build_prompt`).
            system_prompt: Contenu du system prompt, deja charge par l'appelant.
            config: Parametres numeriques du pipeline (k, top_n, budget).
        """
        self._guardrail_input = guardrail_input
        self._retriever = retriever
        self._reranker = reranker
        self._selector = selector
        self._llm = llm
        self._guardrail_output = guardrail_output
        self._token_counter = token_counter
        self._assemble_prompt = assemble_prompt
        self._system_prompt = system_prompt
        self._config = config

    def run(self, question: str) -> Answer:
        """Execute la chaine complete pour une question et produit une `Answer`.

        Args:
            question: La question de l'utilisateur.

        Returns:
            La reponse produite, avec `abstained=True` si la question est
            rejetee par le guardrail d'entree, si aucun chunk pertinent ne
            tient dans le budget, si la reponse est rejetee par le guardrail
            de sortie, ou si le LLM abstient explicitement.
        """
        start = time.perf_counter()

        ok, reason = self._guardrail_input.check(question)
        if not ok:
            return self._abstain(reason, stage="guardrail_input", start=start)

        retrieved = self._retriever.retrieve(question, self._config.retrieve_k)
        reranked = self._reranker.rerank(question, retrieved, self._config.rerank_top_n)

        context_budget = (
            self._config.budget_total
            - self._token_counter.count(self._system_prompt)
            - self._token_counter.count(question)
            - self._config.reserve_answer
        )

        if context_budget <= 0:
            return self._abstain(
                ABSTENTION_TEXT,
                stage="budget_exhausted",
                start=start,
                retrieved=retrieved,
                reranked=reranked,
                meta={"context_budget": context_budget},
            )

        selected = self._selector.select(question, reranked, context_budget)

        if not selected:
            return self._abstain(
                ABSTENTION_TEXT,
                stage="no_chunk_selected",
                start=start,
                retrieved=retrieved,
                reranked=reranked,
                meta={"context_budget": context_budget, "n_retrieved": len(retrieved)},
            )

        prompt = self._assemble_prompt(selected, question)
        raw_answer = self._llm.generate(prompt)

        ok, reason = self._guardrail_output.check(raw_answer)
        if not ok:
            return self._abstain(
                reason,
                stage="guardrail_output",
                start=start,
                retrieved=retrieved,
                reranked=reranked,
                selected=selected,
                meta={"context_budget": context_budget},
            )

        sources = sorted({chunk.source for chunk in selected})
        tokens_used = (
            self._token_counter.count(self._system_prompt)
            + self._token_counter.count(prompt)
            + self._token_counter.count(raw_answer)
        )
        reranked_sorted = sorted(reranked, key=lambda scored: scored.score, reverse=True)
        rerank_rank_of = {scored.chunk.id: rank for rank, scored in enumerate(reranked_sorted, start=1)}
        selected_ids_set = {chunk.id for chunk in selected}

        # Liste complète des chunks reranked avec scores + indication utilisé/non
        reranked_chunks_details = [
            {
                "chunk_id": scored.chunk.id,
                "source": scored.chunk.source,
                "score": round(scored.score, 6),
                "n_tokens": scored.chunk.n_tokens,
                "rank": rerank_rank_of[scored.chunk.id],
                "used": scored.chunk.id in selected_ids_set,
                "text_preview": scored.chunk.text[:100] + "..." if len(scored.chunk.text) > 100 else scored.chunk.text,
            }
            for scored in reranked_sorted
        ]

        return Answer(
            text=raw_answer,
            sources=sources,
            selected_chunk_ids=[chunk.id for chunk in selected],
            tokens_used=tokens_used,
            abstained=ABSTENTION_TEXT in raw_answer,
            meta={
                "context_budget": context_budget,
                "n_retrieved": len(retrieved),
                "n_reranked": len(reranked),
                "n_selected": len(selected),
                "latency_seconds": round(time.perf_counter() - start, 4),
                "selected_chunk_texts": [chunk.text for chunk in selected],
                "selected_chunk_ranks": [rerank_rank_of[chunk.id] for chunk in selected],
                "retrieved_sources": [scored.chunk.source for scored in retrieved],
                "reranked_sources": [scored.chunk.source for scored in reranked_sorted],
                "selected_sources": [chunk.source for chunk in selected],
                "reranked_chunks_details": reranked_chunks_details,  # NOUVEAU
            },
        )

    def _abstain(
        self,
        text: str,
        stage: str,
        start: float,
        retrieved: list[Any] | None = None,
        reranked: list[Any] | None = None,
        selected: list[Chunk] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Answer:
        """Construit une `Answer` d'abstention, quel que soit le point d'arret.

        Args:
            retrieved: Candidats du retriever au moment de l'abstention, s'il
                a deja tourne (`list[ScoredChunk]`), sinon `None`.
            reranked: Candidats reordonnes par le reranker, s'il a deja
                tourne (`list[ScoredChunk]`), sinon `None`.
        """
        retrieved = retrieved or []
        reranked = reranked or []
        selected = selected or []
        return Answer(
            text=text,
            sources=sorted({chunk.source for chunk in selected}),
            selected_chunk_ids=[chunk.id for chunk in selected],
            tokens_used=0,
            abstained=True,
            meta={
                "stage": stage,
                "latency_seconds": round(time.perf_counter() - start, 4),
                "selected_chunk_texts": [chunk.text for chunk in selected],
                "retrieved_sources": [scored.chunk.source for scored in retrieved],
                "reranked_sources": [scored.chunk.source for scored in reranked],
                "selected_sources": [chunk.source for chunk in selected],
                **(meta or {}),
            },
        )
