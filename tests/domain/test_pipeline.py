"""Tests de Pipeline (domain/pipeline.py) : orchestration, budget, abstention.

Aucun mock. Pour la vitesse, le retrieval utilise BM25 (pas de dependance a
BGE-M3/Chroma : la qualite du retrieval est deja testee dans sa propre
brique) et le reranking est desactive (`NoOpReranker`, une vraie strategie
du projet, pas une simulation). Le chemin nominal complet appelle un vrai
serveur Ollama (qwen3:1.7b).
"""

from pathlib import Path

import tiktoken

from src.adapters.guardrails.basic import BasicGuardrail
from src.adapters.llm.ollama import build_prompt, create_llm
from src.adapters.llm.token_counter import TiktokenTokenCounter
from src.adapters.reranking.noop import NoOpReranker
from src.adapters.retrieval.bm25 import BM25Retriever
from src.domain.budget import TopKSelector
from src.domain.models import Chunk
from src.domain.pipeline import ABSTENTION_TEXT, Pipeline, PipelineConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SYSTEM_PROMPT = (_REPO_ROOT / "src" / "prompts" / "system.txt").read_text(encoding="utf-8")
# verifie : 273 tokens cl100k_base (system prompt condense pour recuperer du
# budget de contexte, cf. fix recall q1 : le prompt avait gonfle a 453
# tokens a force d'ajouter des regles, ce qui affamait le budget disponible
# pour les chunks pertinents sur les questions multi-documents)

_CORPUS = [
    Chunk(
        id="c1",
        text="The propulsion system uses xenon as fuel for the ion thrusters.",
        source="file01",
        n_tokens=12,
    ),
    Chunk(
        id="c2",
        text="The cafeteria menu today offers pasta and a green salad.",
        source="file02",
        n_tokens=11,
    ),
]


def _build_pipeline(
    guardrail_input_max_length: int = 2000,
    guardrail_output_max_length: int = 4000,
    budget_total: int = 1024,
    reserve_answer: int = 200,
) -> Pipeline:
    token_counter = TiktokenTokenCounter(tiktoken.get_encoding("cl100k_base"))
    retriever = BM25Retriever(chunks=_CORPUS, k1=1.5, b=0.75)
    llm = create_llm(
        {"model": "qwen3:1.7b", "temperature": 0.1},
        system_prompt=_SYSTEM_PROMPT,
        max_output_tokens=reserve_answer,
    )

    return Pipeline(
        guardrail_input=BasicGuardrail(max_length=guardrail_input_max_length),
        retriever=retriever,
        reranker=NoOpReranker(),
        selector=TopKSelector(),
        llm=llm,
        guardrail_output=BasicGuardrail(max_length=guardrail_output_max_length),
        token_counter=token_counter,
        assemble_prompt=build_prompt,
        system_prompt=_SYSTEM_PROMPT,
        config=PipelineConfig(
            retrieve_k=5, rerank_top_n=5, budget_total=budget_total, reserve_answer=reserve_answer
        ),
    )


class TestPipelineGuardrailInput:
    def test_empty_question_abstains_without_calling_llm(self) -> None:
        pipeline = _build_pipeline()
        answer = pipeline.run("")

        assert answer.abstained is True
        assert answer.meta["stage"] == "guardrail_input"
        assert answer.selected_chunk_ids == []
        assert answer.tokens_used == 0

    def test_oversized_question_abstains(self) -> None:
        pipeline = _build_pipeline(guardrail_input_max_length=10)
        answer = pipeline.run("a" * 11)

        assert answer.abstained is True
        assert answer.meta["stage"] == "guardrail_input"


class TestPipelineBudget:
    def test_budget_exhausted_abstains_without_calling_llm(self) -> None:
        # system prompt seul (273 tokens) > budget_total : context_budget <= 0 d'office
        pipeline = _build_pipeline(budget_total=10, reserve_answer=5)
        answer = pipeline.run("What fuels the thrusters?")

        assert answer.abstained is True
        assert answer.meta["stage"] == "budget_exhausted"
        assert answer.meta["context_budget"] <= 0

    def test_selection_never_exceeds_a_tight_positive_budget(self) -> None:
        # calibre pour que context_budget = 306 - 273(system) - 8(question) - 10(reserve) = 15 :
        # assez pour 1 chunk (~12 tokens) mais pas les 2 (~23 tokens).
        pipeline = _build_pipeline(budget_total=306, reserve_answer=10)
        answer = pipeline.run("What fuel does the propulsion system use?")

        assert answer.meta["context_budget"] == 15
        assert len(answer.selected_chunk_ids) <= 1


class TestPipelineHappyPath:
    def test_grounded_question_returns_real_answer_with_sources(self) -> None:
        pipeline = _build_pipeline()
        answer = pipeline.run("What fuel does the propulsion system use?")

        assert answer.abstained is False
        assert "xenon" in answer.text.lower()
        assert "file01" in answer.sources
        assert "c1" in answer.selected_chunk_ids
        assert answer.tokens_used > 0
        assert answer.meta["n_retrieved"] > 0

    def test_selected_chunk_ranks_expose_reranking_order(self) -> None:
        """`meta["selected_chunk_ranks"]` (affiche "Top N" dans Streamlit)
        doit donner le rang (1 = le mieux classe par le reranking) de chaque
        chunk retenu, sans doublon et borne par le nombre de candidats
        reranked."""
        pipeline = _build_pipeline()
        answer = pipeline.run("What fuel does the propulsion system use?")

        ranks = answer.meta["selected_chunk_ranks"]
        assert len(ranks) == len(answer.selected_chunk_ids)
        assert len(set(ranks)) == len(ranks)  # pas de doublon
        assert all(1 <= rank <= answer.meta["n_reranked"] for rank in ranks)

    def test_unanswerable_question_abstains_via_llm(self) -> None:
        pipeline = _build_pipeline()
        answer = pipeline.run("What is the capital of France?")

        assert answer.abstained is True
        assert ABSTENTION_TEXT in answer.text


class TestPipelineGuardrailOutput:
    def test_tiny_output_guardrail_rejects_generated_answer(self) -> None:
        pipeline = _build_pipeline(guardrail_output_max_length=1)
        answer = pipeline.run("What fuel does the propulsion system use?")

        assert answer.abstained is True
        assert answer.meta["stage"] == "guardrail_output"
