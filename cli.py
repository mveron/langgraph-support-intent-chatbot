from collections.abc import Callable
from typing import Any

from graph import build_graph
from llm import OllamaTextModel, OllamaUnavailableError
from runner import stream_graph


def run_cli(
    graph: Any,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    output_fn("LangGraph + Ollama demo. Type 'exit' to quit.")

    while True:
        question = input_fn("\nQuestion: ").strip()
        if question.lower() in {"", "exit"}:
            return

        try:
            events = list(stream_graph(graph, question))
        except OllamaUnavailableError as exc:
            output_fn(f"Error: {exc}")
            continue

        final = events[-1].state
        route = " -> ".join(event.node for event in events)
        output_fn(f"Route: {route}")
        output_fn(f"Category: {final['category']}")
        output_fn(f"Answer: {final['answer']}")


def main() -> None:
    run_cli(build_graph(OllamaTextModel()))


if __name__ == "__main__":
    main()
