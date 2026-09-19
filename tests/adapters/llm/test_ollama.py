"""Tests de l'adapter Ollama (port LLM) : build_prompt, wiring config, system prompt.

Aucun test n'appelle un serveur Ollama reel : ni ce sandbox ni la session de
build n'ont Ollama installe/demarre, et aucun modele n'est encore telecharge
(cf. les commandes `ollama pull` fournies a l'utilisateur). Simuler cet appel
reseau irait a l'encontre de la consigne "aucun mock" plutot que la respecter :
on ne teste donc que ce qui est reellement executable ici (assemblage du
prompt, lecture de la config, contenu du system prompt), et le test de
`OllamaLLM.generate()` en conditions reelles reste a faire par l'utilisateur
une fois les modeles telecharges (voir le resume de reponse).
"""

from pathlib import Path

import ollama
import pytest
import tiktoken
import yaml

from src.adapters.llm.ollama import OllamaLLM, build_prompt, create_llm
from src.adapters.llm.token_counter import TiktokenTokenCounter
from src.domain.budget import TopKSelector
from src.domain.models import Chunk, ScoredChunk

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SYSTEM_PROMPT_PATH = _REPO_ROOT / "src" / "prompts" / "system.txt"
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "default.yaml"


def _chunk(chunk_id: str, source: str, text: str, n_tokens: int) -> Chunk:
    return Chunk(id=chunk_id, text=text, source=source, n_tokens=n_tokens)


class TestBuildPrompt:
    def test_prompt_contains_sources_in_citation_format(self) -> None:
        chunks = [
            _chunk("c1", "file01", "Xenon fuels the ion thrusters.", 10),
            _chunk("c2", "file02", "Boot sequence initializes core services.", 10),
        ]

        prompt = build_prompt(chunks, "What fuels the thrusters?")

        assert "[Source: file01]" in prompt
        assert "[Source: file02]" in prompt

    def test_prompt_contains_question(self) -> None:
        prompt = build_prompt([], "What fuels the thrusters?")
        assert "What fuels the thrusters?" in prompt

    def test_prompt_numbers_chunks_in_order(self) -> None:
        chunks = [
            _chunk("c1", "file01", "first", 5),
            _chunk("c2", "file02", "second", 5),
        ]
        prompt = build_prompt(chunks, "q")

        assert prompt.index("Chunk 1") < prompt.index("Chunk 2")
        assert prompt.index("Chunk 1") < prompt.index("[Source: file01]")

    def test_empty_chunks_still_contains_question(self) -> None:
        prompt = build_prompt([], "orphan question")
        assert "orphan question" in prompt


class TestPromptStaysUnderBudget:
    """Le contexte assemble reste sous context_budget : garanti par le Selector
    en amont (domain/budget.py), demontre ici de bout en bout avec un vrai
    TokenCounter (tiktoken) sur un texte reel."""

    def test_selected_chunks_fit_budget_and_appear_in_prompt(self) -> None:
        counter = TiktokenTokenCounter(tiktoken.get_encoding("cl100k_base"))

        texts = {
            "file01": "The propulsion system uses xenon as fuel for the ion thrusters.",
            "file02": "The cafeteria menu today offers pasta and a green salad.",
            "file03": "Safety procedures list all emergency exits located on deck two.",
        }
        scored_chunks = [
            ScoredChunk(
                chunk=_chunk(source, source, text, counter.count(text)), score=score
            )
            for (source, text), score in zip(texts.items(), (0.9, 0.5, 0.3))
        ]

        context_budget = counter.count(texts["file01"]) + counter.count(texts["file02"])
        selected = TopKSelector().select("q", scored_chunks, context_budget)

        assert sum(c.n_tokens for c in selected) <= context_budget

        prompt = build_prompt(selected, "What fuels the thrusters?")
        for chunk in selected:
            assert f"[Source: {chunk.source}]" in prompt


class TestSystemPrompt:
    def test_contains_abstention_sentence(self) -> None:
        text = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        assert "Information non trouvée dans les documents." in text

    def test_contains_citation_format_instruction(self) -> None:
        text = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        assert "[Source: fileNN]" in text

    def test_mentions_contradictions(self) -> None:
        text = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        assert "contradict" in text.lower()


class TestCreateLLM:
    def test_reads_model_and_temperature_from_config(self) -> None:
        llm = create_llm(
            {"model": "qwen3:1.7b", "temperature": 0.1}, system_prompt="system prompt text"
        )

        assert isinstance(llm, OllamaLLM)
        assert llm._model == "qwen3:1.7b"
        assert llm._temperature == 0.1
        assert llm._system_prompt == "system prompt text"
        assert isinstance(llm._client, ollama.Client)

    def test_reads_default_config_model_and_temperature(self) -> None:
        config = yaml.safe_load(_DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        llm = create_llm(config["llm"], system_prompt="x")

        assert llm._model == "qwen3:1.7b"
        assert llm._temperature == 0.1

    def test_missing_model_raises(self) -> None:
        with pytest.raises(KeyError):
            create_llm({"temperature": 0.1}, system_prompt="x")

    def test_missing_temperature_raises(self) -> None:
        with pytest.raises(KeyError):
            create_llm({"model": "qwen3:1.7b"}, system_prompt="x")


class TestExperimentConfigsOnlyChangeModel:
    """Chaque config/experiments/llm_*.yaml ne doit contenir que llm.model."""

    @pytest.mark.parametrize(
        ("filename", "expected_model"),
        [
            ("llm_smollm.yaml", "smollm2:1.7b"),
            ("llm_qwen3.yaml", "qwen3:1.7b"),
            ("llm_qwen35.yaml", "qwen3.5:2b"),
        ],
    )
    def test_only_overrides_llm_model(self, filename: str, expected_model: str) -> None:
        path = _REPO_ROOT / "config" / "experiments" / filename
        override = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert set(override.keys()) == {"llm"}
        assert set(override["llm"].keys()) == {"model"}
        assert override["llm"]["model"] == expected_model
