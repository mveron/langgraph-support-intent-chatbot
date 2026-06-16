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
    output_fn("Support chatbot powered by LangGraph + Ollama. Type 'exit' to quit.")
    conversation_history: list[dict[str, str]] = []

    while True:
        message = input_fn("\nSupport message: ").strip()
        if message.lower() in {"", "exit"}:
            return

        try:
            events = list(
                stream_graph(
                    graph,
                    message,
                    conversation_history=conversation_history,
                )
            )
        except OllamaUnavailableError as exc:
            output_fn(f"Error: {exc}")
            continue

        final = events[-1].state
        route = " -> ".join(event.node for event in events)
        output_fn(f"Route: {route}")
        output_fn(f"Category: {final['category']}")
        output_fn(f"Ticket action: {final.get('ticket_action', 'none')}")
        output_fn(f"Queue: {final.get('ticket_queue', 'none')}")
        output_fn(f"Priority: {final.get('ticket_priority', 'none')}")
        output_fn(f"Answer: {final['answer']}")
        conversation_history.append({"role": "user", "content": message})
        conversation_history.append({"role": "assistant", "content": final["answer"]})


def main() -> None:
    run_cli(build_graph(OllamaTextModel()))


if __name__ == "__main__":
    main()
