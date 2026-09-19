"""Tests de RequestLogger : une ligne JSON structuree par requete (regle CLAUDE.md)."""

import json
import logging

import pytest

from src.domain.models import Answer
from src.observability.logging import RequestLogger

_ANSWER = Answer(
    text="The propulsion system uses xenon [Source: file01].",
    sources=["file01"],
    selected_chunk_ids=["c1"],
    tokens_used=42,
    abstained=False,
    meta={"context_budget": 800, "n_selected": 1},
)


class TestRequestLogger:
    def test_logs_one_json_line_with_required_fields(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        test_logger = logging.getLogger("test_request_logger")
        logger = RequestLogger(test_logger)

        with caplog.at_level(logging.INFO, logger="test_request_logger"):
            logger.log_request(
                question="What fuel does it use?", answer=_ANSWER, latency_seconds=1.23456
            )

        assert len(caplog.records) == 1
        event = json.loads(caplog.records[0].message)

        assert event["question"] == "What fuel does it use?"
        assert event["answer"] == _ANSWER.text
        assert event["sources"] == ["file01"]
        assert event["selected_chunk_ids"] == ["c1"]
        assert event["tokens_used"] == 42
        assert event["abstained"] is False
        assert event["latency_seconds"] == 1.2346  # arrondi a 4 decimales
        assert event["meta"] == {"context_budget": 800, "n_selected": 1}

    def test_log_line_is_valid_json(self, caplog: pytest.LogCaptureFixture) -> None:
        test_logger = logging.getLogger("test_request_logger_json")
        logger = RequestLogger(test_logger)

        with caplog.at_level(logging.INFO, logger="test_request_logger_json"):
            logger.log_request(question="q", answer=_ANSWER, latency_seconds=0.1)

        json.loads(caplog.records[0].message)  # ne doit pas lever
