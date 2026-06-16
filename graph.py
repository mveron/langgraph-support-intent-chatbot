import operator
from typing import Annotated, Literal, Protocol

from typing_extensions import Required, TypedDict

from langgraph.graph import END, START, StateGraph


Category = Literal["billing", "technical", "account", "general"]


class TextModel(Protocol):
    def generate(self, prompt: str) -> str: ...


class GraphState(TypedDict, total=False):
    message: Required[str]
    category: Category
    answer: str
    trace: Annotated[list[str], operator.add]


class GraphUpdate(TypedDict, total=False):
    category: Category
    answer: str
    trace: list[str]


def normalize_category(raw: str) -> Category:
    normalized = raw.strip().lower()
    if normalized in {"billing", "technical", "account"}:
        return normalized
    return "general"


def build_graph(model: TextModel):
    def classify_ticket(state: GraphState) -> GraphUpdate:
        prompt = (
            "Classify this support message as billing, technical, account, "
            "or general. Answer with exactly one word from that list.\n"
            f"Support message: {state['message']}\n/no_think"
        )
        category = normalize_category(model.generate(prompt))
        return {
            "category": category,
            "trace": [f"classify_ticket: category={category}"],
        }

    def billing_support(state: GraphState) -> GraphUpdate:
        prompt = (
            "You are a billing support specialist. Acknowledge the issue, ask "
            "for invoice or payment details if needed, and explain the next step.\n"
            f"Support message: {state['message']}\n/no_think"
        )
        return {
            "answer": model.generate(prompt).strip(),
            "trace": ["billing_support: answer generated"],
        }

    def technical_support(state: GraphState) -> GraphUpdate:
        prompt = (
            "You are a technical support specialist. Ask for the error message, "
            "environment, and reproduction steps when useful.\n"
            f"Support message: {state['message']}\n/no_think"
        )
        return {
            "answer": model.generate(prompt).strip(),
            "trace": ["technical_support: answer generated"],
        }

    def account_support(state: GraphState) -> GraphUpdate:
        prompt = (
            "You are an account support specialist. Help with login, password, "
            "profile, access, or verification issues without requesting secrets.\n"
            f"Support message: {state['message']}\n/no_think"
        )
        return {
            "answer": model.generate(prompt).strip(),
            "trace": ["account_support: answer generated"],
        }

    def general_support(state: GraphState) -> GraphUpdate:
        prompt = (
            "You are a support triage assistant. Give a helpful first response "
            "and ask one clarifying question if the issue is unclear.\n"
            f"Support message: {state['message']}\n/no_think"
        )
        return {
            "answer": model.generate(prompt).strip(),
            "trace": ["general_support: answer generated"],
        }

    def route(state: GraphState) -> Category:
        return state["category"]

    builder = StateGraph(GraphState)
    builder.add_node("classify_ticket", classify_ticket)
    builder.add_node("billing_support", billing_support)
    builder.add_node("technical_support", technical_support)
    builder.add_node("account_support", account_support)
    builder.add_node("general_support", general_support)
    builder.add_edge(START, "classify_ticket")
    builder.add_conditional_edges(
        "classify_ticket",
        route,
        {
            "billing": "billing_support",
            "technical": "technical_support",
            "account": "account_support",
            "general": "general_support",
        },
    )
    builder.add_edge("billing_support", END)
    builder.add_edge("technical_support", END)
    builder.add_edge("account_support", END)
    builder.add_edge("general_support", END)
    return builder.compile()
