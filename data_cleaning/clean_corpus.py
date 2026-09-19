"""Nettoie les fichiers .md du corpus (data/docs/) : encodage, espaces, fins de ligne.

Diagnostic (verifie sur le corpus reel) : certains fichiers sont enregistres
en cp1252 et non en UTF-8 valide (leurs guillemets/tirets typographiques
deviennent illisibles si on les lit comme de l'UTF-8) ; tous ont des fins de
ligne CRLF.

Corrige :
- Encodage : decode en UTF-8, avec repli sur cp1252 si invalide, puis
  reecrit toujours en UTF-8 propre.
- Caracteres typographiques (guillemets courbes, tirets cadratins/demi-
  cadratins) -> normalises en ASCII simple.
- Fins de ligne CRLF/CR -> LF.
- Espaces multiples en milieu de ligne, espaces en fin de ligne, lignes
  vides en exces (3+ consecutives) -> supprimes.

Modifie les fichiers ORIGINAUX de data/docs/ directement, en place. Comme
ces fichiers ne sont pas forcement suivis par git au moment du nettoyage,
une copie de chaque fichier modifie est d'abord sauvegardee dans
data_cleaning/originals_backup/ (filet de securite, non versionne).

Usage:
    uv run python data_cleaning/clean_corpus.py
"""

import re
import shutil
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOCS_DIR = _REPO_ROOT / "data" / "docs"
_BACKUP_DIR = Path(__file__).resolve().parent / "originals_backup"

_TYPOGRAPHIC_REPLACEMENTS = {
    "“": '"',  # “ guillemet courbe ouvrant
    "”": '"',  # ” guillemet courbe fermant
    "‘": "'",  # ‘ apostrophe courbe ouvrante
    "’": "'",  # ’ apostrophe courbe fermante
    "–": "-",  # – tiret demi-cadratin
    "—": "-",  # — tiret cadratin
}


def _decode(raw: bytes) -> str:
    """Decode les octets en UTF-8, avec repli sur cp1252 si le fichier n'est
    pas de l'UTF-8 valide (cas observe sur une partie du corpus)."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def clean_text(text: str) -> str:
    """Normalise le texte : fins de ligne, caracteres typographiques, espaces."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    for special, replacement in _TYPOGRAPHIC_REPLACEMENTS.items():
        text = text.replace(special, replacement)

    lines = []
    for line in text.split("\n"):
        line = line.rstrip()
        # espaces multiples en milieu de ligne (garde l'indentation en debut de ligne)
        line = re.sub(r"(?<=\S) {2,}", " ", line)
        lines.append(line)
    text = "\n".join(lines)

    text = re.sub(r"\n{3,}", "\n\n", text)  # au plus une ligne vide consecutive
    text = text.strip("\n") + "\n"

    return text


def clean_file(path: Path, backup_dir: Path) -> bool:
    """Nettoie un fichier en place (sauvegarde d'abord l'original si modifie).

    Args:
        path: Fichier a nettoyer, modifie en place.
        backup_dir: Dossier ou copier l'original avant modification (cree si
            besoin, seulement si le fichier est effectivement modifie).

    Returns:
        True si le contenu du fichier a change.
    """
    raw = path.read_bytes()
    cleaned_text = clean_text(_decode(raw))
    new_bytes = cleaned_text.encode("utf-8")

    if new_bytes == raw:
        return False

    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_dir / path.name)

    path.write_bytes(new_bytes)
    return True


def main() -> None:
    md_files = sorted(_DOCS_DIR.glob("*.md"))
    if not md_files:
        print(f"Aucun fichier .md trouve dans {_DOCS_DIR}")
        return

    changed = 0
    for path in md_files:
        if clean_file(path, backup_dir=_BACKUP_DIR):
            changed += 1
            print(f"nettoye     : {path.name}")
        else:
            print(f"deja propre : {path.name}")

    print(f"\n{changed}/{len(md_files)} fichier(s) modifie(s).")
    if changed:
        print(f"Originaux sauvegardes dans {_BACKUP_DIR}")


if __name__ == "__main__":
    main()
