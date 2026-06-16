from graph import build_graph, normalize_category


class FakeModel:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return next(self.responses)


def test_routes_technical_question_to_technical_node():
    model = FakeModel(["technical", "Use a StateGraph."])
    graph = build_graph(model)

    result = graph.invoke({"question": "How does LangGraph work?", "trace": []})

    assert result["category"] == "technical"
    assert result["answer"] == "Use a StateGraph."
    assert result["trace"] == [
        "classify: category=technical",
        "answer_technical: answer generated",
    ]
    assert len(model.prompts) == 2
    assert "How does LangGraph work?" in model.prompts[0]
    assert "/no_think" in model.prompts[0]
    assert "technical specialist" in model.prompts[1].lower()
    assert "How does LangGraph work?" in model.prompts[1]
    assert "/no_think" in model.prompts[1]


def test_routes_general_question_to_general_node():
    model = FakeModel(["general", "That is a good question."])
    graph = build_graph(model)

    result = graph.invoke({"question": "How are you?", "trace": []})

    assert result["category"] == "general"
    assert result["answer"] == "That is a good question."
    assert result["trace"] == [
        "classify: category=general",
        "answer_general: answer generated",
    ]
    assert len(model.prompts) == 2
    assert "general assistant" in model.prompts[1].lower()
    assert "How are you?" in model.prompts[1]
    assert "/no_think" in model.prompts[1]


def test_unknown_classification_falls_back_to_general():
    model = FakeModel(["not sure", "General answer."])
    graph = build_graph(model)

    result = graph.invoke({"question": "Ambiguous question", "trace": []})

    assert result["category"] == "general"
    assert result["answer"] == "General answer."
    assert result["trace"][-1] == "answer_general: answer generated"
    assert "general assistant" in model.prompts[1].lower()


def test_normalizes_case_and_whitespace_for_technical_category():
    assert normalize_category("  TECHNICAL\n") == "technical"


def test_input_schema_requires_question():
    graph = build_graph(FakeModel([]))

    schema = graph.get_input_jsonschema()

    assert schema["required"] == ["question"]
