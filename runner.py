from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from graph import ConversationMessage, GraphState, GraphUpdate


@dataclass(frozen=True)
class StepEvent:
    node: str
    update: GraphUpdate
    state: GraphState


def _snapshot_state(state: GraphState) -> GraphState:
    snapshot: GraphState = dict(state)
    if "trace" in snapshot:
        snapshot["trace"] = list(snapshot["trace"])
    if "conversation_history" in snapshot:
        snapshot["conversation_history"] = [
            dict(message) for message in snapshot["conversation_history"]
        ]
    if "ticket_database" in snapshot:
        snapshot["ticket_database"] = deepcopy(snapshot["ticket_database"])
    if "ticket_record" in snapshot:
        snapshot["ticket_record"] = dict(snapshot["ticket_record"])
    return snapshot


def stream_graph(
    graph: Any,
    message: str,
    conversation_history: list[ConversationMessage] | None = None,
) -> Iterator[StepEvent]:
    state: GraphState = {
        "message": message,
        "conversation_history": [dict(item) for item in conversation_history or []],
        "trace": [],
    }

    for chunk in graph.stream(state, stream_mode="updates"):
        for node, update in chunk.items():
            for key, value in update.items():
                if key == "trace":
                    state["trace"] = state.get("trace", []) + value
                else:
                    state[key] = value
            yield StepEvent(node=node, update=update, state=_snapshot_state(state))


def run_graph(
    graph: Any,
    message: str,
    conversation_history: list[ConversationMessage] | None = None,
) -> GraphState:
    state: GraphState = {
        "message": message,
        "conversation_history": [dict(item) for item in conversation_history or []],
        "trace": [],
    }

    for event in stream_graph(graph, message, conversation_history=conversation_history):
        state = event.state

    return state
