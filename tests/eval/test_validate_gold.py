"""Tests de eval/validate_gold.py : detection d'incoherences, sans jamais
ecrire dans le gold set ni le corpus (verifie explicitement)."""

from pathlib import Path

import pytest

from eval.validate_gold import GoldValidationError, validate_gold


def _write_gold(path: Path, content: str) -> Path:
    gold_path = path / "gold.yaml"
    gold_path.write_text(content, encoding="utf-8")
    return gold_path


def _write_corpus(path: Path, filenames: list[str]) -> Path:
    corpus_dir = path / "docs"
    corpus_dir.mkdir()
    for name in filenames:
        (corpus_dir / name).write_text("contenu", encoding="utf-8")
    return corpus_dir


class TestValidGoldSet:
    def test_passes_silently_when_consistent(self, tmp_path: Path) -> None:
        gold = _write_gold(
            tmp_path,
            "- id: q1\n  question: \"q\"\n  relevant_sources: [a.md]\n  key_points: [\"fait 1\"]\n",
        )
        corpus = _write_corpus(tmp_path, ["a.md"])

        validate_gold(gold, corpus)  # ne leve rien

    def test_key_points_optional(self, tmp_path: Path) -> None:
        gold = _write_gold(tmp_path, "- id: q1\n  question: \"q\"\n  relevant_sources: [a.md]\n")
        corpus = _write_corpus(tmp_path, ["a.md"])

        validate_gold(gold, corpus)


class TestEmptyRelevantSources:
    def test_raises_with_item_id(self, tmp_path: Path) -> None:
        gold = _write_gold(tmp_path, "- id: q1\n  question: \"q\"\n  relevant_sources: []\n")
        corpus = _write_corpus(tmp_path, [])

        with pytest.raises(GoldValidationError, match="q1.*relevant_sources.*vide"):
            validate_gold(gold, corpus)


class TestSourceMissingFromCorpus:
    def test_raises_naming_the_missing_file(self, tmp_path: Path) -> None:
        gold = _write_gold(tmp_path, "- id: q1\n  question: \"q\"\n  relevant_sources: [ghost.md]\n")
        corpus = _write_corpus(tmp_path, ["a.md"])

        with pytest.raises(GoldValidationError, match="ghost.md"):
            validate_gold(gold, corpus)


class TestEmptyKeyPoint:
    def test_raises_for_blank_key_point(self, tmp_path: Path) -> None:
        gold = _write_gold(
            tmp_path,
            "- id: q1\n  question: \"q\"\n  relevant_sources: [a.md]\n  key_points: [\"\"]\n",
        )
        corpus = _write_corpus(tmp_path, ["a.md"])

        with pytest.raises(GoldValidationError, match="key_point"):
            validate_gold(gold, corpus)

    def test_raises_for_whitespace_only_key_point(self, tmp_path: Path) -> None:
        gold = _write_gold(
            tmp_path,
            "- id: q1\n  question: \"q\"\n  relevant_sources: [a.md]\n  key_points: [\"   \"]\n",
        )
        corpus = _write_corpus(tmp_path, ["a.md"])

        with pytest.raises(GoldValidationError, match="key_point"):
            validate_gold(gold, corpus)


class TestMultipleErrorsReportedTogether:
    def test_lists_every_problem_not_just_the_first(self, tmp_path: Path) -> None:
        gold = _write_gold(
            tmp_path,
            "- id: q1\n  question: \"q\"\n  relevant_sources: [ghost.md]\n  key_points: [\"\"]\n",
        )
        corpus = _write_corpus(tmp_path, [])

        with pytest.raises(GoldValidationError) as exc_info:
            validate_gold(gold, corpus)

        message = str(exc_info.value)
        assert "ghost.md" in message
        assert "key_point" in message


class TestEmptyGoldFile:
    def test_raises(self, tmp_path: Path) -> None:
        gold = _write_gold(tmp_path, "")
        corpus = _write_corpus(tmp_path, [])

        with pytest.raises(GoldValidationError):
            validate_gold(gold, corpus)


class TestNeverWrites:
    def test_gold_file_content_unchanged_after_validation(self, tmp_path: Path) -> None:
        content = "- id: q1\n  question: \"q\"\n  relevant_sources: [a.md]\n"
        gold = _write_gold(tmp_path, content)
        corpus = _write_corpus(tmp_path, ["a.md"])

        validate_gold(gold, corpus)

        assert gold.read_text(encoding="utf-8") == content

    def test_real_gold_file_is_valid(self) -> None:
        """Le vrai eval/gold_retrieval.yaml (IMMUTABLE) doit passer sans erreur."""
        validate_gold()
