"""Tests de application/answer.py : build_pipeline (fail loud) et
answer_question (delegation au Pipeline + journalisation structuree).

La couverture "bout en bout reel" (build_pipeline avec tous les adapters
lourds) est dans tests/test_cli.py ; ici on verifie specifiquement le
comportement propre a cette couche (erreurs de config, journalisation).
"""

import logging
from pathlib import Path

import pytest
import tiktoken

from src.adapters.guardrails.basic import BasicGuardrail
from src.adapters.llm.ollama import build_prompt, create_llm
from src.adapters.llm.token_counter import TiktokenTokenCounter
from src.adapters.reranking.noop import NoOpReranker
from src.adapters.retrieval.bm25 import BM25Retriever
from src.application.answer import answer_question, build_pipeline
from src.domain.budget import TopKSelector
from src.domain.models import Chunk
from src.domain.pipeline import Pipeline, PipelineConfig
from src.observability.logging import RequestLogger

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SYSTEM_PROMPT = (_REPO_ROOT / "src" / "prompts" / "system.txt").read_text(encoding="utf-8")

_CORPUS = [
    Chunk(
        id="c1",
        text="The propulsion system uses xenon as fuel for the ion thrusters.",
        source="file01",
        n_tokens=12,
    ),
]


class TestBuildPipeline:
    def test_missing_config_section_raises(self) -> None:
        with pytest.raises(KeyError):
            build_pipeline({}, chunks=[], system_prompt="x")


class TestAnswerQuestion:
    def test_delegates_to_pipeline_and_logs_the_request(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        token_counter = TiktokenTokenCounter(tiktoken.get_encoding("cl100k_base"))
        pipeline = Pipeline(
            guardrail_input=BasicGuardrail(max_length=2000),
            retriever=BM25Retriever(chunks=_CORPUS, k1=1.5, b=0.75),
            reranker=NoOpReranker(),
            selector=TopKSelector(),
            llm=create_llm(
                {"model": "qwen3:1.7b", "temperature": 0.1},
                system_prompt=_SYSTEM_PROMPT,
                max_output_tokens=200,
            ),
            guardrail_output=BasicGuardrail(max_length=4000),
            token_counter=token_counter,
            assemble_prompt=build_prompt,
            system_prompt=_SYSTEM_PROMPT,
            config=PipelineConfig(
                retrieve_k=5, rerank_top_n=5, budget_total=1024, reserve_answer=200
            ),
        )
        test_logger = logging.getLogger("test_answer_question")
        logger = RequestLogger(test_logger)

        with caplog.at_level(logging.INFO, logger="test_answer_question"):
            answer = answer_question(
                "What fuel does the propulsion system use?", pipeline, logger
            )

        assert answer.abstained is False
        assert "xenon" in answer.text.lower()

        assert len(caplog.records) == 1
        logged = caplog.records[0].message
        assert "xenon" in logged.lower()
        assert '"question"' in logged
        assert '"tokens_used"' in logged
