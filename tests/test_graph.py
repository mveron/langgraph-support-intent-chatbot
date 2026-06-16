from graph import build_graph, normalize_category
from ticket_db import (
    build_mock_ticket_record,
    extract_ticket_id,
    format_ticket_status,
    next_ticket_id,
    parse_ticket_database,
)


class FakeModel:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return next(self.responses)


def test_routes_billing_issue_to_billing_support_node():
    model = FakeModel(["billing", "I can help review that charge."])
    graph = build_graph(model)

    result = graph.invoke(
        {
            "message": "I was charged twice this month.",
            "conversation_history": [],
            "trace": [],
        }
    )

    assert result["category"] == "billing"
    assert result["answer"].startswith(
        "I created support ticket TCK-1004 in the billing_operations queue "
        "with high priority."
    )
    assert "I can help review that charge." in result["answer"]
    assert result["trace"] == [
        "prepare_context: reason captured",
        "classify_ticket: category=billing",
        "assess_ticket_need: action=create_ticket",
        "load_ticket_database: loaded 3 tickets",
        "create_ticket: ticket_id=TCK-1004",
        "billing_support: answer generated",
    ]
    assert result["ticket_action"] == "create_ticket"
    assert result["ticket_id"] == "TCK-1004"
    assert result["support_reason"] == "I was charged twice this month."
    assert result["ticket_queue"] == "billing_operations"
    assert result["ticket_priority"] == "high"
    assert result["ticket_record"]["status"] == "Open"
    assert result["ticket_record"]["queue"] == "billing_operations"
    assert result["ticket_record"]["priority"] == "high"
    assert result["ticket_record"]["reason"] == "I was charged twice this month."
    assert len(model.prompts) == 2
    assert "I was charged twice this month." in model.prompts[0]
    assert "/no_think" in model.prompts[0]
    assert "billing support specialist" in model.prompts[1].lower()
    assert "I was charged twice this month." in model.prompts[1]
    assert "Queue: billing_operations" in model.prompts[1]
    assert "Priority: high" in model.prompts[1]
    assert "/no_think" in model.prompts[1]


def test_routes_technical_issue_to_technical_support_node():
    model = FakeModel(["technical", "Please share the error message."])
    graph = build_graph(model)

    result = graph.invoke(
        {
            "message": "The app crashes on startup.",
            "conversation_history": [],
            "trace": [],
        }
    )

    assert result["category"] == "technical"
    assert result["trace"] == [
        "prepare_context: reason captured",
        "classify_ticket: category=technical",
        "assess_ticket_need: action=create_ticket",
        "load_ticket_database: loaded 3 tickets",
        "create_ticket: ticket_id=TCK-1004",
        "technical_support: answer generated",
    ]
    assert result["ticket_queue"] == "technical_support_tier_2"
    assert result["ticket_priority"] == "high"
    assert "technical support specialist" in model.prompts[1].lower()


def test_routes_account_issue_to_account_support_node():
    model = FakeModel(["account", "Let's recover your account."])
    graph = build_graph(model)

    result = graph.invoke(
        {"message": "I cannot log in.", "conversation_history": [], "trace": []}
    )

    assert result["category"] == "account"
    assert result["trace"][-1] == "account_support: answer generated"
    assert result["ticket_action"] == "create_ticket"
    assert result["ticket_queue"] == "identity_access"
    assert "account support specialist" in model.prompts[1].lower()


def test_routes_general_issue_to_general_support_node():
    model = FakeModel(["general", "I can help with that."])
    graph = build_graph(model)

    result = graph.invoke(
        {
            "message": "What can this service do?",
            "conversation_history": [],
            "trace": [],
        }
    )

    assert result["category"] == "general"
    assert result["answer"] == "I can help with that."
    assert result["trace"] == [
        "prepare_context: reason captured",
        "classify_ticket: category=general",
        "general_support: answer generated",
    ]
    assert len(model.prompts) == 2
    assert "support triage assistant" in model.prompts[1].lower()
    assert "What can this service do?" in model.prompts[1]
    assert "/no_think" in model.prompts[1]


def test_unknown_classification_falls_back_to_general():
    model = FakeModel(["not sure", "General answer."])
    graph = build_graph(model)

    result = graph.invoke(
        {"message": "Ambiguous issue", "conversation_history": [], "trace": []}
    )

    assert result["category"] == "general"
    assert result["answer"] == "General answer."
    assert result["trace"][-1] == "general_support: answer generated"
    assert result["trace"][0] == "prepare_context: reason captured"
    assert "support triage assistant" in model.prompts[1].lower()


def test_informational_billing_question_does_not_create_ticket():
    model = FakeModel(["billing", "You can download invoices from settings."])
    graph = build_graph(model)

    result = graph.invoke(
        {
            "message": "How do I download my invoice?",
            "conversation_history": [],
            "trace": [],
        }
    )

    assert result["ticket_action"] == "respond_only"
    assert "ticket_id" not in result
    assert result["support_reason"] == "How do I download my invoice?"
    assert result["ticket_queue"] == "none"
    assert result["trace"] == [
        "prepare_context: reason captured",
        "classify_ticket: category=billing",
        "assess_ticket_need: action=respond_only",
        "billing_support: answer generated",
    ]


def test_normalizes_case_and_whitespace_for_billing_category():
    assert normalize_category("  BILLING\n") == "billing"


def test_normalizes_ticket_status_with_punctuation():
    assert normalize_category("Ticket_Status.\n") == "ticket_status"


def test_classifier_prompt_includes_conversation_history():
    model = FakeModel(["ticket_status", "unused"])
    graph = build_graph(model)

    graph.invoke(
        {
            "message": "What about it?",
            "conversation_history": [
                {"role": "user", "content": "My ticket is TCK-1002."},
                {
                    "role": "assistant",
                    "content": "I can check that if you ask for status.",
                },
            ],
            "trace": [],
        }
    )

    assert "Conversation history:" in model.prompts[0]
    assert "user: My ticket is TCK-1002." in model.prompts[0]
    assert "Current message: What about it?" in model.prompts[0]


def test_prepare_context_uses_history_for_short_follow_up():
    model = FakeModel(["technical", "Please send logs."])
    graph = build_graph(model)

    result = graph.invoke(
        {
            "message": "It still fails.",
            "conversation_history": [
                {
                    "role": "user",
                    "content": "The report export crashes with a 500 error.",
                },
                {
                    "role": "assistant",
                    "content": "Please retry and tell me if it continues.",
                },
            ],
            "trace": [],
        }
    )

    assert result["support_reason"] == (
        "Follow-up: It still fails. Previous issue: "
        "The report export crashes with a 500 error."
    )
    assert result["ticket_action"] == "create_ticket"
    assert result["ticket_queue"] == "technical_support_tier_2"


def test_routes_ticket_status_query_to_mock_ticket_lookup():
    model = FakeModel(["ticket_status"])
    graph = build_graph(model)

    result = graph.invoke(
        {
            "message": "Can you check TCK-1002?",
            "conversation_history": [],
            "trace": [],
        }
    )

    assert result["category"] == "ticket_status"
    assert result["ticket_id"] == "TCK-1002"
    assert "Waiting on Customer" in result["answer"]
    assert "Duplicate invoice charge" in result["answer"]
    assert result["trace"] == [
        "prepare_context: reason captured",
        "classify_ticket: category=ticket_status",
        "load_ticket_database: loaded 3 tickets",
        "lookup_ticket_status: ticket_id=TCK-1002",
        "ticket_status_response: answer generated",
    ]


def test_non_ticket_routes_do_not_load_ticket_database():
    model = FakeModel(["general", "General answer"])
    graph = build_graph(model)

    result = graph.invoke(
        {
            "message": "What can this service do?",
            "conversation_history": [],
            "trace": [],
        }
    )

    assert "load_ticket_database: loaded 3 tickets" not in result["trace"]


def test_created_ticket_context_is_added_to_support_prompt():
    model = FakeModel(["technical", "Please send logs."])
    graph = build_graph(model)

    result = graph.invoke(
        {
            "message": "The app crashes when I open reports.",
            "conversation_history": [],
            "trace": [],
        }
    )

    assert "Created ticket: TCK-1004" in model.prompts[1]
    assert result["answer"].startswith("I created support ticket TCK-1004")
    assert "Queue: technical_support_tier_2" in model.prompts[1]
    assert "Decision: create_ticket" in model.prompts[1]


def test_ticket_status_uses_ticket_id_from_history_when_follow_up_is_ambiguous():
    model = FakeModel(["ticket_status"])
    graph = build_graph(model)

    result = graph.invoke(
        {
            "message": "What is the latest status?",
            "conversation_history": [
                {"role": "user", "content": "My ticket number is TCK-1003."},
            ],
            "trace": [],
        }
    )

    assert result["ticket_id"] == "TCK-1003"
    assert "Resolved" in result["answer"]


def test_ticket_status_asks_for_ticket_id_when_missing():
    model = FakeModel(["ticket_status"])
    graph = build_graph(model)

    result = graph.invoke(
        {
            "message": "Can you check my ticket?",
            "conversation_history": [],
            "trace": [],
        }
    )

    assert "Please share your ticket ID" in result["answer"]
    assert result["trace"][-2:] == [
        "lookup_ticket_status: ticket_id=missing",
        "ticket_status_response: answer generated",
    ]


def test_extract_ticket_id_accepts_prefixed_and_plain_ticket_numbers():
    assert extract_ticket_id("Please check TCK-1002") == "TCK-1002"
    assert extract_ticket_id("ticket 1003") == "TCK-1003"


def test_parse_ticket_database_and_format_status():
    tickets = parse_ticket_database(
        "TCK-1002 | customer: Contoso | status: Waiting on Customer | "
        "summary: Duplicate invoice charge | owner: Billing | updated: 2026-06-14"
    )

    assert tickets["TCK-1002"]["status"] == "Waiting on Customer"
    assert "Contoso" in format_ticket_status("TCK-1002", tickets["TCK-1002"])


def test_next_ticket_id_uses_highest_existing_mock_ticket_number():
    tickets = {
        "TCK-1002": {"status": "Open"},
        "TCK-1010": {"status": "Resolved"},
    }

    assert next_ticket_id(tickets) == "TCK-1011"


def test_build_mock_ticket_record_uses_category_owner_and_summary():
    record = build_mock_ticket_record(
        category="technical",
        message="The app crashes when exporting reports.",
        queue="technical_support_tier_2",
        priority="high",
        reason="Reports export crashes.",
    )

    assert record["status"] == "Open"
    assert record["owner"] == "technical_support_tier_2"
    assert record["queue"] == "technical_support_tier_2"
    assert record["priority"] == "high"
    assert record["reason"] == "Reports export crashes."
    assert record["summary"] == "Reports export crashes."


def test_input_schema_requires_message():
    graph = build_graph(FakeModel([]))

    schema = graph.get_input_jsonschema()

    assert schema["required"] == ["message"]
