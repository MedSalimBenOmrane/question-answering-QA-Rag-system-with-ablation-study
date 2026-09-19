"""Tests de data_cleaning/clean_corpus.py : encodage, typographie, espaces.

Les cas reproduits ici viennent du diagnostic reel effectue sur le corpus
(data/docs/) : certains fichiers en cp1252 (guillemets/tirets illisibles en
UTF-8), CRLF partout, espaces multiples, lignes vides en exces.
"""

from pathlib import Path

from data_cleaning.clean_corpus import _decode, clean_file, clean_text


class TestDecode:
    def test_decodes_valid_utf8(self) -> None:
        assert _decode("café".encode("utf-8")) == "café"

    def test_falls_back_to_cp1252_for_invalid_utf8(self) -> None:
        # 0x93/0x94 en cp1252 = guillemets courbes ; invalide en UTF-8 seul.
        raw = b'\x93Safe Mode\x94'
        assert _decode(raw) == "“Safe Mode”"


class TestCleanText:
    def test_normalizes_crlf_to_lf(self) -> None:
        assert clean_text("line1\r\nline2\r\n") == "line1\nline2\n"

    def test_normalizes_curly_quotes_and_dashes_to_ascii(self) -> None:
        text = "“Safe Mode” – also ‘isolation’ — mode"
        result = clean_text(text)
        assert result == '"Safe Mode" - also \'isolation\' - mode\n'

    def test_strips_trailing_whitespace_per_line(self) -> None:
        assert clean_text("line with trailing spaces   \nother\n") == (
            "line with trailing spaces\nother\n"
        )

    def test_collapses_multiple_internal_spaces(self) -> None:
        assert clean_text("word1    word2  word3\n") == "word1 word2 word3\n"

    def test_preserves_leading_indentation(self) -> None:
        # les espaces de debut de ligne (indentation markdown) ne sont pas touches
        assert clean_text("  - item one\n  - item two\n") == "  - item one\n  - item two\n"

    def test_collapses_excess_blank_lines(self) -> None:
        assert clean_text("a\n\n\n\n\nb\n") == "a\n\nb\n"

    def test_strips_leading_and_trailing_blank_lines(self) -> None:
        assert clean_text("\n\n# Title\n\ncontent\n\n\n") == "# Title\n\ncontent\n"

    def test_idempotent_on_already_clean_text(self) -> None:
        cleaned_once = clean_text("# Title\n\nSome \"quoted\" text - plain.\n")
        assert clean_text(cleaned_once) == cleaned_once

    def test_realistic_mojibake_style_input(self) -> None:
        text = _decode(b'# Ambiguous Uses of \x93Safe Mode\x94\r\n\r\nThe term \x93Safe Mode\x94.\r\n')
        result = clean_text(text)
        assert result == '# Ambiguous Uses of "Safe Mode"\n\nThe term "Safe Mode".\n'


class TestCleanFile:
    def test_cleans_file_in_place_and_backs_up_original(self, tmp_path: Path) -> None:
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        backup_dir = tmp_path / "backup"

        target = docs_dir / "sample.md"
        target.write_bytes(b'# Title\r\n\r\nThe \x93term\x94 has  double   spaces.  \r\n')

        changed = clean_file(target, backup_dir=backup_dir)

        assert changed is True
        assert target.read_text(encoding="utf-8") == (
            '# Title\n\nThe "term" has double spaces.\n'
        )
        assert (backup_dir / "sample.md").exists()
        assert (backup_dir / "sample.md").read_bytes() == (
            b'# Title\r\n\r\nThe \x93term\x94 has  double   spaces.  \r\n'
        )

    def test_already_clean_file_is_left_untouched_and_not_backed_up(
        self, tmp_path: Path
    ) -> None:
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        backup_dir = tmp_path / "backup"

        target = docs_dir / "sample.md"
        clean_content = "# Title\n\nAlready clean.\n"
        # write_bytes plutot que write_text : evite la traduction \n -> \r\n
        # que Python applique en mode texte sur Windows, qui aurait fausse
        # la premisse "deja propre" du test.
        target.write_bytes(clean_content.encode("utf-8"))

        changed = clean_file(target, backup_dir=backup_dir)

        assert changed is False
        assert target.read_bytes() == clean_content.encode("utf-8")
        assert not backup_dir.exists()
