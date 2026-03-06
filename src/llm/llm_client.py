from __future__ import annotations

import os
import requests

from src.ui.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SEC


def generate_response(
    prompt: str,
    model: str | None = None,
    timeout_sec: int | None = None,
    num_predict: int | None = None,
    temperature: float = 0.2,
) -> str:
    model_to_use = model or OLLAMA_MODEL
    endpoint = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"

    t = int(timeout_sec) if timeout_sec is not None else int(OLLAMA_TIMEOUT_SEC)
    n = int(num_predict) if num_predict is not None else int(os.getenv("LLM_NUM_PREDICT", "220"))

    payload = {
        "model": model_to_use,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": n,
        },
    }

    try:
        response = requests.post(endpoint, json=payload, timeout=t)
        response.raise_for_status()
        return response.json().get("response", "") or ""
    except requests.exceptions.Timeout:
        return f"Error: Ollama request timed out after {t}s."
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {e}"