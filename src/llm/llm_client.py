import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


def generate_response(prompt: str, model: str = "llama3.2:3b") -> str:
    """
    Sends a prompt to the local Ollama model
    and returns the generated response.
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {e}"