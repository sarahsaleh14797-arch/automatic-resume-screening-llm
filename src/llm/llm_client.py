from __future__ import annotations

import requests

from src.ui.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SEC


def generate_response(prompt: str, model: str | None = None) -> str:
    model_to_use = model or OLLAMA_MODEL
    endpoint = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {"model": model_to_use, "prompt": prompt, "stream": False}

    try:
        response = requests.post(endpoint, json=payload, timeout=OLLAMA_TIMEOUT_SEC)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.Timeout:
        return f"Error: Ollama request timed out after {OLLAMA_TIMEOUT_SEC}s."
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {e}"