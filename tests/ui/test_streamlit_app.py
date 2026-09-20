"""Tests de src/ui/streamlit_app.py.

`TestStreamlitAppReal` execute reellement le script via
`streamlit.testing.v1.AppTest` (aucun mock) : vrai pipeline (index reel deja
construit via `uv run python -m src.cli index`, vrais modeles BGE-M3 +
cross-encoder, vrai serveur Ollama). Necessite que le corpus soit deja
indexe et Ollama actif.
"""

import json
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.domain.models import Answer
from src.ui.streamlit_app import FEEDBACK_LEVELS, log_feedback

_APP_PATH = str(Path(__file__).resolve().parents[2] / "src" / "ui" / "streamlit_app.py")

_ANSWER = Answer(
    text="The boot halts at phase 3. [Source: boot_sequence.md]",
    sources=["boot_sequence.md"],
    selected_chunk_ids=["boot_sequence.md#0"],
    tokens_used=427,
    abstained=False,
)


class TestLogFeedback:
    def test_writes_one_jsonl_line_with_expected_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "feedback.jsonl"

        log_feedback("What halts the boot?", _ANSWER, "High", path=path)

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["question"] == "What halts the boot?"
        assert event["feedback"] == "High"
        assert event["sources"] == ["boot_sequence.md"]
        assert event["tokens_used"] == 427

    def test_appends_without_overwriting(self, tmp_path: Path) -> None:
        path = tmp_path / "feedback.jsonl"
        log_feedback("q1", _ANSWER, "Poor", path=path)
        log_feedback("q2", _ANSWER, "Very high", path=path)

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["feedback"] == "Poor"
        assert json.loads(lines[1])["feedback"] == "Very high"

    def test_invalid_level_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            log_feedback("q", _ANSWER, "Excellent", path=tmp_path / "f.jsonl")

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "feedback.jsonl"
        log_feedback("q", _ANSWER, "Medium", path=path)
        assert path.exists()


class TestStreamlitAppReal:
    """Execution reelle de l'app (pas de mock) : necessite un index deja
    construit (`uv run python -m src.cli index`) et un serveur Ollama actif."""

    def test_app_loads_without_error(self) -> None:
        at = AppTest.from_file(_APP_PATH, default_timeout=120)
        at.run()

        assert not at.exception
        assert [t.value for t in at.title] == ["Context-Aware QA"]
        assert at.button[0].label == "Poser la question"

    def test_grounded_question_renders_answer_sources_chunks_and_budget(self) -> None:
        at = AppTest.from_file(_APP_PATH, default_timeout=120)
        at.run()

        at.text_input(key="question_input").input(
            "If the QRC fails during boot, what is the expected system behavior "
            "and recommended recovery steps?"
        )
        at.button[0].click()
        at.run()

        assert not at.exception

        answer_texts = [m.value for m in at.markdown]
        assert any("boot" in text.lower() for text in answer_texts)

        # Les sources affichees (st.caption) viennent des metadonnees de
        # retrieval, pas du texte libre du LLM : fiables et deterministes,
        # contrairement a la citation [Source: ...] que le petit modele local
        # (qwen3:1.7b) n'inclut pas de facon fiable dans sa reponse generee
        # (verifie : absente sur certains runs malgre la consigne systeme).
        assert any(re.search(r"\.md\b", c.value) for c in at.caption)
        assert any("Chunk 1" in text for text in answer_texts)
        assert any(len(t.value) > 0 for t in at.text)

        progress = at.get("progress")
        assert len(progress) == 1
        assert progress[0].text.endswith("/ 1024 tokens")

        feedback_labels = {b.label for b in at.button} & set(FEEDBACK_LEVELS)
        assert feedback_labels == set(FEEDBACK_LEVELS)

    def test_feedback_button_click_logs_and_shows_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # AppTest execute le script dans un contexte isole : monkeypatch.setattr
        # sur le module deja importe ne l'atteint pas, mais os.environ (process-
        # global) est bien relu par le module au moment ou AppTest l'execute.
        log_path = tmp_path / "feedback.jsonl"
        monkeypatch.setenv("CONTEXT_AWARE_QA_FEEDBACK_LOG_PATH", str(log_path))

        at = AppTest.from_file(_APP_PATH, default_timeout=120)
        at.run()
        at.text_input(key="question_input").input("What does the cafeteria serve?")
        at.button[0].click()
        at.run()

        high_button = next(b for b in at.button if b.label == "High")
        high_button.click()
        at.run()

        assert not at.exception
        assert any("Feedback enregistre" in s.value for s in at.success)
        assert log_path.exists()
        event = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
        assert event["feedback"] == "High"
