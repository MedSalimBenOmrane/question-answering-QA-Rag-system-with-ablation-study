"""Fixture partagee : vrai TokenCounter aligne sur llm.model (qwen3:1.7b), pas de mock.

Utilise HFTokenCounter avec le vrai tokenizer HuggingFace de Qwen3 (le meme
tokenizer que le modele Ollama cible), pas tiktoken : voir config/default.yaml
(token_counter.provider = hf). TiktokenTokenCounter reste disponible et teste
separement dans tests/adapters/llm/test_token_counter.py.
"""

import pytest
import transformers

from src.adapters.llm.token_counter import HFTokenCounter


@pytest.fixture(scope="session")
def real_token_counter() -> HFTokenCounter:
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    return HFTokenCounter(tokenizer)
