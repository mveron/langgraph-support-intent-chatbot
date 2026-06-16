from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from graph import GraphState, GraphUpdate


@dataclass(frozen=True)
class StepEvent:
    node: str
    update: GraphUpdate
    state: GraphState


def _snapshot_state(state: GraphState) -> GraphState:
    snapshot: GraphState = dict(state)
    if "trace" in snapshot:
        snapshot["trace"] = list(snapshot["trace"])
    return snapshot


def stream_graph(graph: Any, question: str) -> Iterator[StepEvent]:
    state: GraphState = {"question": question, "trace": []}

    for chunk in graph.stream(state, stream_mode="updates"):
        for node, update in chunk.items():
            for key, value in update.items():
                if key == "trace":
                    state["trace"] = state.get("trace", []) + value
                else:
                    state[key] = value
            yield StepEvent(node=node, update=update, state=_snapshot_state(state))


def run_graph(graph: Any, question: str) -> GraphState:
    state: GraphState = {"question": question, "trace": []}

    for event in stream_graph(graph, question):
        state = event.state

    return state
