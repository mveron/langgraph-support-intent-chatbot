from cli import run_cli
from graph import build_graph
from llm import OllamaUnavailableError


class FakeModel:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)

    def generate(self, prompt: str) -> str:
        return next(self.responses)


class UnavailableModel:
    def generate(self, prompt: str) -> str:
        raise OllamaUnavailableError("Ollama is unavailable")


def test_run_cli_streams_general_question_and_prints_final_result():
    graph = build_graph(FakeModel(["general", "General answer."]))
    inputs = iter(["Hello", "exit"])
    outputs: list[str] = []

    run_cli(graph, input_fn=lambda prompt: next(inputs), output_fn=outputs.append)

    output = "\n".join(outputs)
    assert "LangGraph + Ollama demo" in output
    assert "classify -> answer_general" in output
    assert "Category: general" in output
    assert "Answer: General answer." in output


def test_run_cli_reports_ollama_error_and_continues_until_exit():
    graph = build_graph(UnavailableModel())
    inputs = iter(["Hello", "exit"])
    prompts: list[str] = []
    outputs: list[str] = []

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        return next(inputs)

    run_cli(graph, input_fn=input_fn, output_fn=outputs.append)

    assert prompts == ["\nQuestion: ", "\nQuestion: "]
    assert "Error: Ollama is unavailable" in "\n".join(outputs)
