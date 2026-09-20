"""Adapter LLM base sur Ollama (modele local, port `LLM`).

Le modele et la temperature sont LUS DEPUIS LA CONFIG (jamais codes en dur) :
comparer plusieurs LLM ( smollm2:1.7b, qwen3:1.7b, qwen3.5:2b)
revient donc a changer `llm.model` dans la config (voir
`config/experiments/llm_*.yaml`), jamais a toucher ce fichier.

Les instructions de grounding (abstention, citations, contradictions) sont
portees par le system prompt (`src/prompts/system.txt`), transmis a Ollama
via le parametre natif `system` de son API - pas concatenees dans le prompt
utilisateur. Zero-shot : `build_prompt` n'injecte aucun exemple.
"""

from typing import Any

import ollama

from src.domain.models import Chunk


def build_prompt(chunks: list[Chunk], question: str) -> str:
    """Assemble le prompt utilisateur : contexte ordonne (chunks numerotes) + question.

    Args:
        chunks: Chunks a presenter au LLM, dans l'ordre voulu (deja selectionnes
            et ordonnes en amont par le Selector, budget deja respecte).
        question: La question de l'utilisateur.

    Returns:
        Le texte du prompt, avec chaque chunk numerote et attribue a sa source
        au format de citation attendu du LLM (ex: "Chunk 1 [Source: file01]\\n<texte>").
    """
    numbered_context = "\n\n".join(
        f"Chunk {index} [Source: {chunk.source}]\n{chunk.text}"
        for index, chunk in enumerate(chunks, start=1)
    )
    return f"CONTEXT\n{numbered_context}\n\nQUESTION\n{question}\n\nANSWER:"


class OllamaLLM:
    """LLM adapter s'appuyant sur un modele Ollama local."""

    def __init__(
        self,
        client: ollama.Client,
        model: str,
        temperature: float,
        system_prompt: str,
    ) -> None:
        """Initialise l'adapter.

        Args:
            client: Client Ollama deja instancie.
            model: Tag du modele Ollama a utiliser (ex: "qwen3:1.7b").
            temperature: Temperature d'echantillonnage du modele.
            system_prompt: Instructions systeme (grounding, abstention,
                citations) - transmises separement du prompt utilisateur.
        """
        self._client = client
        self._model = model
        self._temperature = temperature
        self._system_prompt = system_prompt

    def generate(self, prompt: str) -> str:
        """Genere une reponse via Ollama.

        Args:
            prompt: Le prompt utilisateur (contexte + question), typiquement
                produit par `build_prompt`.

        Returns:
            Le texte genere par le modele.
        """
        response = self._client.generate(
            model=self._model,
            prompt=prompt,
            system=self._system_prompt,
            options={"temperature": self._temperature},
        )
        return response.response


def create_llm(config: dict[str, Any], system_prompt: str) -> OllamaLLM:
    """Instancie l'adapter Ollama configure.

    Args:
        config: Section de configuration `llm`. Doit contenir `model` et
            `temperature`. `base_url` est optionnel (defaut : Ollama local,
            http://localhost:11434).
        system_prompt: Contenu du system prompt (ex: lu depuis
            `src/prompts/system.txt` par l'appelant).

    Returns:
        Un `OllamaLLM` pret a l'emploi.

    Raises:
        KeyError: Si `model` ou `temperature` sont absents de la config.
    """
    try:
        model = config["model"]
        temperature = config["temperature"]
    except KeyError as exc:
        raise KeyError(
            "config de llm invalide : 'model' et 'temperature' sont requis"
        ) from exc

    client = ollama.Client(host=config.get("base_url"))
    return OllamaLLM(
        client=client, model=model, temperature=temperature, system_prompt=system_prompt
    )
