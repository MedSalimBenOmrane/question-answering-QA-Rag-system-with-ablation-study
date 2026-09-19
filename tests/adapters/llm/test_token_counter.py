"""Tests du TokenCounter concret (tiktoken, HF, factory) : aucun mock.

Chaque test charge le vrai tokenizer (encodage tiktoken reel, tokenizer HF
reel via transformers) et verifie le comptage reel de tokens.
"""

import tiktoken
import transformers
import pytest

from src.adapters.llm.token_counter import (
    HFTokenCounter,
    TiktokenTokenCounter,
    create_token_counter,
)


class TestTiktokenTokenCounter:
    def test_count_matches_real_cl100k_base_encoding(self) -> None:
        encoding = tiktoken.get_encoding("cl100k_base")
        counter = TiktokenTokenCounter(encoding)

        assert counter.count("hello world") == len(encoding.encode("hello world"))
        assert counter.count("hello world") > 0

    def test_count_is_zero_for_empty_text(self) -> None:
        counter = TiktokenTokenCounter(tiktoken.get_encoding("cl100k_base"))
        assert counter.count("") == 0


class TestHFTokenCounter:
    def test_count_matches_real_gpt2_tokenizer(self) -> None:
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        counter = HFTokenCounter(tokenizer)

        assert counter.count("hello world") == len(tokenizer.encode("hello world"))
        assert counter.count("hello world") > 0

    def test_count_is_zero_for_empty_text(self) -> None:
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        counter = HFTokenCounter(tokenizer)
        assert counter.count("") == 0


class TestCreateTokenCounter:
    def test_dispatches_to_tiktoken_and_counts_for_real(self) -> None:
        counter = create_token_counter(
            {"provider": "tiktoken", "tiktoken": {"encoding": "cl100k_base"}}
        )
        assert isinstance(counter, TiktokenTokenCounter)

        encoding = tiktoken.get_encoding("cl100k_base")
        assert counter.count("hello world") == len(encoding.encode("hello world"))

    def test_dispatches_to_hf_and_counts_for_real(self) -> None:
        counter = create_token_counter({"provider": "hf", "hf": {"model_name": "gpt2"}})
        assert isinstance(counter, HFTokenCounter)

        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        assert counter.count("hello world") == len(tokenizer.encode("hello world"))

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
