"""Tests du BasicGuardrail (texte vide / trop long) et de sa factory."""

import pytest

from src.adapters.guardrails.basic import BasicGuardrail, create_guardrail


class TestBasicGuardrail:
    def test_accepts_normal_text(self) -> None:
        guardrail = BasicGuardrail(max_length=100)
        ok, reason = guardrail.check("What fuels the thrusters?")
        assert ok is True
        assert reason == ""

    def test_rejects_empty_text(self) -> None:
        guardrail = BasicGuardrail(max_length=100)
        ok, reason = guardrail.check("")
        assert ok is False
        assert reason != ""

    def test_rejects_whitespace_only_text(self) -> None:
        guardrail = BasicGuardrail(max_length=100)
        ok, _ = guardrail.check("   \n\t  ")
        assert ok is False

    def test_rejects_text_over_max_length(self) -> None:
        guardrail = BasicGuardrail(max_length=10)
        ok, reason = guardrail.check("a" * 11)
        assert ok is False
        assert "10" in reason

    def test_accepts_text_at_exactly_max_length(self) -> None:
        guardrail = BasicGuardrail(max_length=10)
        ok, _ = guardrail.check("a" * 10)
        assert ok is True


class TestCreateGuardrail:
    def test_creates_from_config(self) -> None:
        guardrail = create_guardrail({"max_length": 50})
        assert isinstance(guardrail, BasicGuardrail)
        ok, _ = guardrail.check("a" * 51)
        assert ok is False

    def test_missing_max_length_raises(self) -> None:
        with pytest.raises(KeyError):
            create_guardrail({})
