from typing import Any

from langchain_ollama import ChatOllama


MODEL_NAME = "qwen3:4b"


class OllamaUnavailableError(RuntimeError):
    pass


class OllamaTextModel:
    def __init__(self, client: Any | None = None):
        self.client = (
            client
            if client is not None
            else ChatOllama(model=MODEL_NAME, temperature=0)
        )

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.invoke(prompt)
            return str(response.content)
        except Exception as exc:
            raise OllamaUnavailableError(
                f"Could not use Ollama with the {MODEL_NAME} model. "
                "Verify that Ollama is installed, that the service is running, "
                f"and that the {MODEL_NAME} model is available locally."
            ) from exc
