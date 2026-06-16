import pytest

from llm import OllamaTextModel, OllamaUnavailableError


class FakeResponse:
    content = "local response"


class FakeClient:
    def __init__(self):
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        return FakeResponse()


class BrokenClient:
    def __init__(self):
        self.error = ConnectionError("connection refused")

    def invoke(self, prompt: str):
        raise self.error


class FalseyClient(FakeClient):
    def __bool__(self) -> bool:
        return False


class BrokenContentResponse:
    @property
    def content(self):
        raise ValueError("bad content")


class BrokenContentClient:
    def invoke(self, prompt: str) -> BrokenContentResponse:
        return BrokenContentResponse()


class FallbackClient(FakeClient):
    def invoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        response = FakeResponse()
        response.content = "fallback"
        return response


def test_generate_returns_text_from_injected_client():
    client = FakeClient()
    model = OllamaTextModel(client=client)

    result = model.generate("hola")

    assert result == "local response"
    assert client.prompts == ["hola"]


def test_init_uses_injected_client_even_when_falsey(monkeypatch):
    fallback_client = FallbackClient()
    monkeypatch.setattr("llm.ChatOllama", lambda **kwargs: fallback_client)
    client = FalseyClient()
    model = OllamaTextModel(client=client)

    result = model.generate("hola")

    assert result == "local response"
    assert client.prompts == ["hola"]
    assert fallback_client.prompts == []


def test_generate_translates_invoke_failure_to_ollama_unavailable_error():
    model = OllamaTextModel(client=BrokenClient())

    with pytest.raises(OllamaUnavailableError) as exc_info:
        model.generate("hola")

    message = str(exc_info.value)
    assert "Ollama" in message
    assert "qwen3:4b" in message
    assert "service" in message
    assert "installed" in message


def test_generate_preserves_original_exception_as_cause():
    client = BrokenClient()
    model = OllamaTextModel(client=client)

    with pytest.raises(OllamaUnavailableError) as exc_info:
        model.generate("hola")

    assert exc_info.value.__cause__ is client.error


def test_generate_translates_content_access_failure_and_preserves_cause():
    model = OllamaTextModel(client=BrokenContentClient())

    with pytest.raises(OllamaUnavailableError) as exc_info:
        model.generate("hola")

    assert isinstance(exc_info.value.__cause__, ValueError)
