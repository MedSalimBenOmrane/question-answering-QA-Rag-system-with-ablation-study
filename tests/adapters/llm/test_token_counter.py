"""Tests du TokenCounter concret (tiktoken, HF, factory) : aucun mock.

Chaque test charge le vrai tokenizer (encodage tiktoken reel, tokenizer HF
reel via transformers) et verifie le comptage reel de tokens.
"""

from pathlib import Path

import tiktoken
import transformers
import pytest
import yaml

from src.adapters.llm.token_counter import (
    HFTokenCounter,
    TiktokenTokenCounter,
    create_token_counter,
)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"


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


class TestDefaultConfigMatchesLLMModel:
    """Le TokenCounter actif par defaut (config/default.yaml) doit tokenizer
    exactement comme llm.model (qwen3:1.7b), pas un autre modele/famille."""

    def test_default_provider_is_hf_not_tiktoken(self) -> None:
        config = yaml.safe_load(_DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        assert config["token_counter"]["provider"] == "hf"

    def test_default_hf_model_matches_llm_model_family(self) -> None:
        config = yaml.safe_load(_DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))

        llm_model = config["llm"]["model"]
        hf_model_name = config["token_counter"]["hf"]["model_name"]

        assert llm_model == "qwen3:1.7b"
        # Qwen3 reutilise le tokenizer Qwen2 (meme famille/vocab que le tag Ollama ci-dessus).
        assert hf_model_name == "Qwen/Qwen3-1.7B"

    def test_active_default_counter_matches_known_text_with_real_qwen3_tokenizer(self) -> None:
        config = yaml.safe_load(_DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))

        counter = create_token_counter(config["token_counter"])
        assert isinstance(counter, HFTokenCounter)

        reference_tokenizer = transformers.AutoTokenizer.from_pretrained(
            config["token_counter"]["hf"]["model_name"]
        )
        text = "The propulsion system uses xenon as fuel for the ion thrusters."

        assert counter.count(text) == len(reference_tokenizer.encode(text))
        assert counter.count(text) == 14
