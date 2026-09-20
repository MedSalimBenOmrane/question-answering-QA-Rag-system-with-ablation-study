"""Interface Streamlit du systeme RAG.

Question -> reponse citee, section "chunks utilises", jauge de budget de
tokens (tokens_used / budget.total), et feedback utilisateur a 4 niveaux
(Poor/Medium/High/Very high), journalise en JSON (une ligne par feedback).

L'index et les modeles (embedder, cross-encoder, LLM, etc., via le Pipeline
complet) sont charges une seule fois par processus serveur grace a
`@st.cache_resource` : rechargement evite a chaque interaction utilisateur.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# `streamlit run` execute ce fichier comme script principal : Python ajoute
# le dossier du script (src/ui/) a sys.path, jamais la racine du projet,
# quel que soit le repertoire courant depuis lequel la commande est lancee.
# Sans cette ligne, `from src...` echoue toujours avec
# "ModuleNotFoundError: No module named 'src'" (verifie en direct, y compris
# en lancant depuis la racine).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from src.application.answer import answer_question, build_pipeline
from src.cli import load_chunks, load_config, load_system_prompt
from src.domain.models import Answer
from src.domain.pipeline import Pipeline
from src.observability.logging import RequestLogger
# Surchargeable via CONTEXT_AWARE_QA_FEEDBACK_LOG_PATH (tests d'integration :
# AppTest execute le script dans un contexte isole que monkeypatch.setattr ne
# peut pas atteindre, mais qui lit bien os.environ a chaque execution).
_FEEDBACK_LOG_PATH = Path(
    os.environ.get("CONTEXT_AWARE_QA_FEEDBACK_LOG_PATH", str(_REPO_ROOT / "data" / "feedback.jsonl"))
)

FEEDBACK_LEVELS = ("Poor", "Medium", "High", "Very high")


@st.cache_data
def get_config() -> dict[str, Any]:
    """Charge la configuration (donnee simple, mise en cache par valeur)."""
    return load_config()


@st.cache_resource
def get_pipeline() -> Pipeline:
    """Construit le Pipeline complet (index + modeles) une seule fois par serveur.

    Returns:
        Le `Pipeline` pret a repondre aux questions.

    Raises:
        FileNotFoundError: Si le corpus n'a pas encore ete indexe.
    """
    config = get_config()

    chunks_path = Path(config["vectorstore"]["chunks_path"])
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Index introuvable ({chunks_path}). Executez "
            "`uv run python -m src.cli index` avant de lancer l'application."
        )
    chunks = load_chunks(chunks_path)
    system_prompt = load_system_prompt()

    return build_pipeline(config, chunks, system_prompt)


def log_feedback(question: str, answer: Answer, level: str, path: Path = _FEEDBACK_LOG_PATH) -> None:
    """Journalise un feedback utilisateur en JSON (une ligne par feedback, append-only).

    Args:
        question: La question posee.
        answer: La reponse jugee par l'utilisateur.
        level: Le niveau de feedback choisi (cf. `FEEDBACK_LEVELS`).
        path: Fichier de log (JSON Lines).

    Raises:
        ValueError: Si `level` n'est pas un niveau de feedback valide.
    """
    if level not in FEEDBACK_LEVELS:
        raise ValueError(f"niveau de feedback invalide: {level!r} (attendu parmi {FEEDBACK_LEVELS})")

    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "question": question,
        "answer": answer.text,
        "sources": answer.sources,
        "selected_chunk_ids": answer.selected_chunk_ids,
        "tokens_used": answer.tokens_used,
        "abstained": answer.abstained,
        "feedback": level,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def render_answer(question: str, answer: Answer, budget_total: int) -> None:
    """Affiche la reponse, ses sources, les chunks utilises, la jauge de budget
    et les boutons de feedback."""
    st.subheader("Reponse")
    st.write(answer.text)

    if answer.sources:
        st.caption("Sources : " + ", ".join(answer.sources))

    chunk_ids = answer.selected_chunk_ids
    chunk_texts = answer.meta.get("selected_chunk_texts", [])
    with st.expander(f"Chunks utilises ({len(chunk_ids)})"):
        if not chunk_ids:
            st.write("Aucun chunk retenu.")
        for index, (chunk_id, text) in enumerate(zip(chunk_ids, chunk_texts), start=1):
            st.markdown(f"**Chunk {index}** (`{chunk_id}`)")
            st.text(text)

    st.subheader("Budget de tokens")
    fraction = min(answer.tokens_used / budget_total, 1.0) if budget_total > 0 else 0.0
    st.progress(fraction, text=f"{answer.tokens_used} / {budget_total} tokens")

    st.subheader("Cette reponse etait...")
    columns = st.columns(len(FEEDBACK_LEVELS))
    for column, level in zip(columns, FEEDBACK_LEVELS):
        if column.button(level, key=f"feedback_{level}_{question}"):
            log_feedback(question, answer, level)
            st.success(f"Feedback enregistre : {level}")


def main() -> None:
    """Point d'entree de l'application Streamlit."""
    st.set_page_config(page_title="Context-Aware QA", page_icon="🔎")
    st.title("Context-Aware QA")

    try:
        pipeline = get_pipeline()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    config = get_config()
    logger = RequestLogger()

    question = st.text_input("Question", key="question_input")
    ask_clicked = st.button("Poser la question")

    if ask_clicked and question.strip():
        with st.spinner("Recherche en cours..."):
            answer = answer_question(question, pipeline, logger)
        st.session_state["last_question"] = question
        st.session_state["last_answer"] = answer

    if "last_answer" in st.session_state:
        render_answer(
            st.session_state["last_question"],
            st.session_state["last_answer"],
            config["budget"]["total"],
        )


main()
