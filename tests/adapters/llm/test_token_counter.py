"""Tests du TokenCounter concret (tiktoken, HF, factory)."""

import pytest

from src.adapters.llm.token_counter import (
    HFTokenCounter,
    TiktokenTokenCounter,
    create_token_counter,
)


class FakeEncoder:
    """Encodeur factice : 1 token = 1 mot, pour tester le wrapper hors reseau."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


class TestTiktokenTokenCounter:
    def test_count_with_fake_encoder(self) -> None:
        counter = TiktokenTokenCounter(FakeEncoder())
        assert counter.count("un deux trois") == 3

    def test_count_with_real_cl100k_base(self) -> None:
        tiktoken = pytest.importorskip("tiktoken")
        encoding = tiktoken.get_encoding("cl100k_base")
        counter = TiktokenTokenCounter(encoding)
        assert counter.count("hello world") == len(encoding.encode("hello world"))
        assert counter.count("hello world") > 0


class TestHFTokenCounter:
    def test_count_with_fake_tokenizer(self) -> None:
        counter = HFTokenCounter(FakeEncoder())
        assert counter.count("un deux trois quatre") == 4

    def test_count_with_real_gpt2_tokenizer(self) -> None:
        transformers = pytest.importorskip("transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        counter = HFTokenCounter(tokenizer)
        assert counter.count("hello world") == len(tokenizer.encode("hello world"))
        assert counter.count("hello world") > 0


class TestCreateTokenCounter:
    def test_dispatches_to_tiktoken(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "tiktoken.get_encoding", lambda name: FakeEncoder(), raising=False
        )
        counter = create_token_counter({"provider": "tiktoken", "tiktoken": {"encoding": "cl100k_base"}})
        assert isinstance(counter, TiktokenTokenCounter)
        assert counter.count("a b c") == 3

    def test_dispatches_to_hf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained",
            lambda name: FakeEncoder(),
            raising=False,
        )
        counter = create_token_counter({"provider": "hf", "hf": {"model_name": "gpt2"}})
        assert isinstance(counter, HFTokenCounter)
        assert counter.count("a b c") == 3

    def test_missing_provider_raises(self) -> None:
        with pytest.raises(KeyError):
            create_token_counter({})

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError):
            create_token_counter({"provider": "does-not-exist"})

    def test_missing_tiktoken_encoding_raises(self) -> None:
        with pytest.raises(KeyError):
            create_token_counter({"provider": "tiktoken", "tiktoken": {}})

    def test_missing_hf_model_name_raises(self) -> None:
        with pytest.raises(KeyError):
            create_token_counter({"provider": "hf", "hf": {}})
