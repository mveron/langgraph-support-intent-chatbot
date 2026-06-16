from graph import build_graph
from runner import run_graph, stream_graph


class FakeModel:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)

    def generate(self, prompt: str) -> str:
        return next(self.responses)


def test_stream_graph_yields_execution_events_with_accumulated_state():
    graph = build_graph(FakeModel(["billing", "Answer"]))

    events = list(stream_graph(graph, "I was charged twice"))

    assert [event.node for event in events] == ["classify_ticket", "billing_support"]
    assert events[0].state["category"] == "billing"
    assert events[0].state["trace"] == ["classify_ticket: category=billing"]
    assert events[-1].state["answer"] == "Answer"
    assert events[-1].state["trace"] == [
        "classify_ticket: category=billing",
        "billing_support: answer generated",
    ]


def test_run_graph_returns_final_state():
    graph = build_graph(FakeModel(["technical", "Answer"]))

    state = run_graph(graph, "The app crashes")

    assert state["answer"] == "Answer"
    assert state["category"] == "technical"
    assert state["trace"] == [
        "classify_ticket: category=technical",
        "technical_support: answer generated",
    ]


def test_stream_graph_events_do_not_share_mutable_trace_state():
    graph = build_graph(FakeModel(["general", "General answer"]))

    events = list(stream_graph(graph, "Can you help?"))
    events[0].state["trace"].append("mutated")

    assert events[1].state["trace"] == [
        "classify_ticket: category=general",
        "general_support: answer generated",
    ]


def test_stream_graph_trace_snapshot_mutation_does_not_contaminate_generator():
    graph = build_graph(FakeModel(["account", "Account answer"]))

    iterator = stream_graph(graph, "I cannot log in")
    first = next(iterator)
    first.state["trace"].append("mutated")
    second = next(iterator)

    assert second.state["trace"] == [
        "classify_ticket: category=account",
        "account_support: answer generated",
    ]
