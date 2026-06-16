from cli import run_cli
from graph import build_graph
from llm import OllamaUnavailableError


class FakeModel:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return next(self.responses)


class UnavailableModel:
    def generate(self, prompt: str) -> str:
        raise OllamaUnavailableError("Ollama is unavailable")


def test_run_cli_streams_general_question_and_prints_final_result():
    graph = build_graph(FakeModel(["billing", "I can help review the charge."]))
    inputs = iter(["I was charged twice", "exit"])
    outputs: list[str] = []

    run_cli(graph, input_fn=lambda prompt: next(inputs), output_fn=outputs.append)

    output = "\n".join(outputs)
    assert "Support chatbot" in output
    assert "Route: classify_ticket -> billing_support" in output
    assert "Category: billing" in output
    assert "Answer: I can help review the charge." in output


def test_run_cli_keeps_history_between_chat_turns():
    model = FakeModel(["ticket_status", "ticket_status"])
    graph = build_graph(model)
    inputs = iter(["Can you check my ticket?", "TCK-1002", "exit"])
    outputs: list[str] = []

    run_cli(graph, input_fn=lambda prompt: next(inputs), output_fn=outputs.append)

    output = "\n".join(outputs)
    assert "Please share your ticket ID" in output
    assert "Waiting on Customer" in output
    assert "Duplicate invoice charge" in output
    assert "Current message: TCK-1002" in model.prompts[1]
    assert "user: Can you check my ticket?" in model.prompts[1]
    assert "assistant: I can check that. Please share your ticket ID" in model.prompts[1]


def test_run_cli_reports_ollama_error_and_continues_until_exit():
    graph = build_graph(UnavailableModel())
    inputs = iter(["I cannot log in", "exit"])
    prompts: list[str] = []
    outputs: list[str] = []

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        return next(inputs)

    run_cli(graph, input_fn=input_fn, output_fn=outputs.append)

    assert prompts == ["\nSupport message: ", "\nSupport message: "]
    assert "Error: Ollama is unavailable" in "\n".join(outputs)
