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

    assert [event.node for event in events] == [
        "classify_ticket",
        "assess_ticket_need",
        "load_ticket_database",
        "create_ticket",
        "billing_support",
    ]
    assert events[0].state["category"] == "billing"
    assert events[0].state["trace"] == [
        "classify_ticket: category=billing",
    ]
    assert events[-1].state["answer"] == (
        "I created support ticket TCK-1004 with status Open. Answer"
    )
    assert events[-1].state["ticket_id"] == "TCK-1004"
    assert events[-1].state["trace"] == [
        "classify_ticket: category=billing",
        "assess_ticket_need: action=create_ticket",
        "load_ticket_database: loaded 3 tickets",
        "create_ticket: ticket_id=TCK-1004",
        "billing_support: answer generated",
    ]


def test_run_graph_returns_final_state():
    graph = build_graph(FakeModel(["technical", "Answer"]))

    state = run_graph(graph, "The app crashes")

    assert state["answer"] == "I created support ticket TCK-1004 with status Open. Answer"
    assert state["category"] == "technical"
    assert state["ticket_id"] == "TCK-1004"
    assert state["trace"] == [
        "classify_ticket: category=technical",
        "assess_ticket_need: action=create_ticket",
        "load_ticket_database: loaded 3 tickets",
        "create_ticket: ticket_id=TCK-1004",
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
        "assess_ticket_need: action=create_ticket",
    ]


def test_stream_graph_passes_conversation_history_into_initial_state():
    graph = build_graph(FakeModel(["ticket_status"]))
    history = [{"role": "user", "content": "My ticket is TCK-1002."}]

    events = list(
        stream_graph(graph, "What is the status?", conversation_history=history)
    )

    assert events[0].state["conversation_history"] == history
    assert events[-1].state["ticket_id"] == "TCK-1002"
    assert "Waiting on Customer" in events[-1].state["answer"]


def test_ticket_database_loads_only_after_ticket_status_classification():
    graph = build_graph(FakeModel(["ticket_status"]))

    events = list(stream_graph(graph, "Status for TCK-1002"))

    assert [event.node for event in events] == [
        "classify_ticket",
        "load_ticket_database",
        "lookup_ticket_status",
        "ticket_status_response",
    ]


def test_stream_graph_snapshots_do_not_share_mutable_history_state():
    graph = build_graph(FakeModel(["general", "General answer"]))
    history = [{"role": "user", "content": "Hello"}]

    events = list(stream_graph(graph, "Can you help?", conversation_history=history))
    events[0].state["conversation_history"].append(
        {"role": "assistant", "content": "mutated"}
    )

    assert events[1].state["conversation_history"] == history
