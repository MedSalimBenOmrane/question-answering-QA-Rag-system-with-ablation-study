"""Valide eval/gold_retrieval.yaml (fichier IMMUTABLE) contre le corpus reel.

Ce module n'ECRIT JAMAIS dans le gold set et ne propose aucune correction
automatique : il se contente de detecter et signaler des incoherences avec
un message explicite, pour etre lance en pre-check avant `eval/run_ablation.py`
(chantier "relative_threshold", etape 3.4).

Verifications :
- `relevant_sources` non vide pour chaque item.
- Chaque fichier cite dans `relevant_sources` existe reellement dans le
  corpus (`data/docs/*.md`) - une source fantome invaliderait silencieusement
  toutes les metriques de retrieval de la question concernee.
- Aucun `key_point` n'est une chaine vide.

Usage:
    uv run python -m eval.validate_gold
"""

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_GOLD_PATH = _REPO_ROOT / "eval" / "gold_retrieval.yaml"
_DEFAULT_CORPUS_DIR = _REPO_ROOT / "data" / "docs"


class GoldValidationError(ValueError):
    """Le gold set (IMMUTABLE) est incoherent avec le corpus, ou mal forme."""


def _corpus_filenames(corpus_dir: Path) -> set[str]:
    """Noms de fichiers `.md` presents dans le corpus reel."""
    return {path.name for path in corpus_dir.glob("*.md")}


def validate_gold(
    gold_path: Path = _DEFAULT_GOLD_PATH, corpus_dir: Path = _DEFAULT_CORPUS_DIR
) -> None:
    """Valide le gold set contre le corpus reel. Ne modifie ni l'un ni l'autre.

    Args:
        gold_path: Chemin du fichier YAML du gold set (IMMUTABLE, lecture seule).
        corpus_dir: Dossier contenant le corpus reel (fichiers `*.md`).

    Raises:
        GoldValidationError: Message listant TOUS les problemes trouves (pas
            seulement le premier), si le gold set est vide, mal forme, ou
            incoherent avec le corpus.
    """
    raw: Any = yaml.safe_load(gold_path.read_text(encoding="utf-8"))
    if not raw:
        raise GoldValidationError(f"gold set vide ou introuvable : {gold_path}")

    corpus_files = _corpus_filenames(corpus_dir)
    errors: list[str] = []

    for entry in raw:
        item_id = entry.get("id", "?")

        relevant_sources = entry.get("relevant_sources") or []
        if not relevant_sources:
            errors.append(f"{item_id} : 'relevant_sources' est vide")
        for source in relevant_sources:
            if source not in corpus_files:
                errors.append(
                    f"{item_id} : source {source!r} citee dans relevant_sources "
                    f"est absente du corpus ({corpus_dir})"
                )

        for index, key_point in enumerate(entry.get("key_points") or [], start=1):
            if not str(key_point).strip():
                errors.append(f"{item_id} : key_point #{index} est une chaine vide")

    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise GoldValidationError(
            f"gold set invalide ({gold_path}), {len(errors)} probleme(s) :\n{details}"
        )


def main() -> None:
    """Valide le gold set reel et affiche le resultat (code de sortie non-zero si invalide)."""
    validate_gold()
    print(f"gold set valide ({_DEFAULT_GOLD_PATH}) : coherent avec le corpus, aucun probleme detecte.")


if __name__ == "__main__":
    main()
