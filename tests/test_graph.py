from graph import build_graph, normalize_category


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

    result = graph.invoke({"message": "I was charged twice this month.", "trace": []})

    assert result["category"] == "billing"
    assert result["answer"] == "I can help review that charge."
    assert result["trace"] == [
        "classify_ticket: category=billing",
        "billing_support: answer generated",
    ]
    assert len(model.prompts) == 2
    assert "I was charged twice this month." in model.prompts[0]
    assert "/no_think" in model.prompts[0]
    assert "billing support specialist" in model.prompts[1].lower()
    assert "I was charged twice this month." in model.prompts[1]
    assert "/no_think" in model.prompts[1]


def test_routes_technical_issue_to_technical_support_node():
    model = FakeModel(["technical", "Please share the error message."])
    graph = build_graph(model)

    result = graph.invoke({"message": "The app crashes on startup.", "trace": []})

    assert result["category"] == "technical"
    assert result["trace"] == [
        "classify_ticket: category=technical",
        "technical_support: answer generated",
    ]
    assert "technical support specialist" in model.prompts[1].lower()


def test_routes_account_issue_to_account_support_node():
    model = FakeModel(["account", "Let's recover your account."])
    graph = build_graph(model)

    result = graph.invoke({"message": "I cannot log in.", "trace": []})

    assert result["category"] == "account"
    assert result["trace"][-1] == "account_support: answer generated"
    assert "account support specialist" in model.prompts[1].lower()


def test_routes_general_issue_to_general_support_node():
    model = FakeModel(["general", "I can help with that."])
    graph = build_graph(model)

    result = graph.invoke({"message": "What can this service do?", "trace": []})

    assert result["category"] == "general"
    assert result["answer"] == "I can help with that."
    assert result["trace"] == [
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

    result = graph.invoke({"message": "Ambiguous issue", "trace": []})

    assert result["category"] == "general"
    assert result["answer"] == "General answer."
    assert result["trace"][-1] == "general_support: answer generated"
    assert "support triage assistant" in model.prompts[1].lower()


def test_normalizes_case_and_whitespace_for_billing_category():
    assert normalize_category("  BILLING\n") == "billing"


def test_input_schema_requires_message():
    graph = build_graph(FakeModel([]))

    schema = graph.get_input_jsonschema()

    assert schema["required"] == ["message"]
