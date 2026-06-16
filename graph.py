import operator
from typing import Annotated, Literal, Protocol

from typing_extensions import Required, TypedDict

from langgraph.graph import END, START, StateGraph


Category = Literal["technical", "general"]


class TextModel(Protocol):
    def generate(self, prompt: str) -> str: ...


class GraphState(TypedDict, total=False):
    question: Required[str]
    category: Category
    answer: str
    trace: Annotated[list[str], operator.add]


class GraphUpdate(TypedDict, total=False):
    category: Category
    answer: str
    trace: list[str]


def normalize_category(raw: str) -> Category:
    normalized = raw.strip().lower()
    return "technical" if normalized == "technical" else "general"


def build_graph(model: TextModel):
    def classify(state: GraphState) -> GraphUpdate:
        prompt = (
            "Classify the question as technical or general. "
            "Answer with exactly one word: technical or general.\n"
            f"Question: {state['question']}\n/no_think"
        )
        category = normalize_category(model.generate(prompt))
        return {
            "category": category,
            "trace": [f"classify: category={category}"],
        }

    def answer_technical(state: GraphState) -> GraphUpdate:
        prompt = (
            "Answer as a technical specialist.\n"
            f"Question: {state['question']}\n/no_think"
        )
        return {
            "answer": model.generate(prompt).strip(),
            "trace": ["answer_technical: answer generated"],
        }

    def answer_general(state: GraphState) -> GraphUpdate:
        prompt = (
            "Answer as a general assistant.\n"
            f"Question: {state['question']}\n/no_think"
        )
        return {
            "answer": model.generate(prompt).strip(),
            "trace": ["answer_general: answer generated"],
        }

    def route(state: GraphState) -> Category:
        return state["category"]

    builder = StateGraph(GraphState)
    builder.add_node("classify", classify)
    builder.add_node("answer_technical", answer_technical)
    builder.add_node("answer_general", answer_general)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route,
        {
            "technical": "answer_technical",
            "general": "answer_general",
        },
    )
    builder.add_edge("answer_technical", END)
    builder.add_edge("answer_general", END)
    return builder.compile()
