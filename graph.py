import operator
import re
from typing import Annotated, Literal, Protocol

from typing_extensions import Required, TypedDict

from langgraph.graph import END, START, StateGraph

from ticket_db import (
    TicketDatabase,
    extract_ticket_id,
    format_ticket_status,
    load_ticket_database as load_mock_ticket_database,
)

Category = Literal["billing", "technical", "account", "ticket_status", "general"]


class ConversationMessage(TypedDict):
    role: str
    content: str


class TextModel(Protocol):
    def generate(self, prompt: str) -> str: ...


class GraphState(TypedDict, total=False):
    message: Required[str]
    conversation_history: list[ConversationMessage]
    ticket_database: TicketDatabase
    ticket_id: str
    ticket_record: dict[str, str]
    category: Category
    answer: str
    trace: Annotated[list[str], operator.add]


class GraphUpdate(TypedDict, total=False):
    conversation_history: list[ConversationMessage]
    ticket_database: TicketDatabase
    ticket_id: str
    ticket_record: dict[str, str]
    category: Category
    answer: str
    trace: list[str]


def normalize_category(raw: str) -> Category:
    normalized = re.sub(r"[^a-z_]", "", raw.strip().lower())
    supported: set[Category] = {
        "billing",
        "technical",
        "account",
        "ticket_status",
        "general",
    }
    if normalized in supported:
        return normalized
    lowered = raw.lower()
    for category in supported:
        if category in lowered:
            return category
    return "general"


def format_conversation_history(history: list[ConversationMessage] | None) -> str:
    if not history:
        return "No previous conversation."

    recent_messages = history[-8:]
    return "\n".join(
        f"{message.get('role', 'unknown')}: {message.get('content', '')}"
        for message in recent_messages
    )


def find_ticket_id_in_state(state: GraphState) -> str | None:
    ticket_id = extract_ticket_id(state["message"])
    if ticket_id:
        return ticket_id

    for message in reversed(state.get("conversation_history", [])):
        ticket_id = extract_ticket_id(message.get("content", ""))
        if ticket_id:
            return ticket_id

    return None


def build_graph(model: TextModel):
    def load_ticket_database(state: GraphState) -> GraphUpdate:
        tickets = load_mock_ticket_database()
        return {
            "ticket_database": tickets,
            "trace": [f"load_ticket_database: loaded {len(tickets)} tickets"],
        }

    def classify_ticket(state: GraphState) -> GraphUpdate:
        history = format_conversation_history(state.get("conversation_history"))
        prompt = (
            "Classify the current support message using the conversation history. "
            "Use billing for invoices, payments, charges, or subscriptions. "
            "Use technical for bugs, errors, integrations, performance, or outages. "
            "Use account for login, password, access, profile, or verification. "
            "Use ticket_status when the user asks about an existing ticket, shares "
            "a ticket ID, or follows up on a previous ticket-status request. "
            "Use general only when none of those apply. Answer with exactly one "
            "word from this list: billing, technical, account, ticket_status, general.\n"
            f"Conversation history:\n{history}\n"
            f"Current message: {state['message']}\n/no_think"
        )
        category = normalize_category(model.generate(prompt))
        return {
            "category": category,
            "trace": [f"classify_ticket: category={category}"],
        }

    def billing_support(state: GraphState) -> GraphUpdate:
        history = format_conversation_history(state.get("conversation_history"))
        prompt = (
            "You are a billing support specialist. Acknowledge the issue, ask "
            "for invoice or payment details if needed, and explain the next step.\n"
            f"Conversation history:\n{history}\n"
            f"Current message: {state['message']}\n/no_think"
        )
        return {
            "answer": model.generate(prompt).strip(),
            "trace": ["billing_support: answer generated"],
        }

    def technical_support(state: GraphState) -> GraphUpdate:
        history = format_conversation_history(state.get("conversation_history"))
        prompt = (
            "You are a technical support specialist. Ask for the error message, "
            "environment, and reproduction steps when useful.\n"
            f"Conversation history:\n{history}\n"
            f"Current message: {state['message']}\n/no_think"
        )
        return {
            "answer": model.generate(prompt).strip(),
            "trace": ["technical_support: answer generated"],
        }

    def account_support(state: GraphState) -> GraphUpdate:
        history = format_conversation_history(state.get("conversation_history"))
        prompt = (
            "You are an account support specialist. Help with login, password, "
            "profile, access, or verification issues without requesting secrets.\n"
            f"Conversation history:\n{history}\n"
            f"Current message: {state['message']}\n/no_think"
        )
        return {
            "answer": model.generate(prompt).strip(),
            "trace": ["account_support: answer generated"],
        }

    def general_support(state: GraphState) -> GraphUpdate:
        history = format_conversation_history(state.get("conversation_history"))
        prompt = (
            "You are a support triage assistant. Give a helpful first response "
            "and ask one clarifying question if the issue is unclear.\n"
            f"Conversation history:\n{history}\n"
            f"Current message: {state['message']}\n/no_think"
        )
        return {
            "answer": model.generate(prompt).strip(),
            "trace": ["general_support: answer generated"],
        }

    def lookup_ticket_status(state: GraphState) -> GraphUpdate:
        ticket_id = find_ticket_id_in_state(state)
        if not ticket_id:
            return {
                "trace": ["lookup_ticket_status: ticket_id=missing"],
            }

        ticket = state.get("ticket_database", {}).get(ticket_id)
        update: GraphUpdate = {
            "ticket_id": ticket_id,
            "trace": [f"lookup_ticket_status: ticket_id={ticket_id}"],
        }
        if ticket:
            update["ticket_record"] = ticket
        return update

    def ticket_status_response(state: GraphState) -> GraphUpdate:
        ticket_id = state.get("ticket_id")
        ticket = state.get("ticket_record")

        if not ticket_id:
            answer = (
                "I can check that. Please share your ticket ID "
                "(for example, TCK-1002)."
            )
        elif not ticket:
            answer = (
                f"I could not find {ticket_id} in the mock ticket database. "
                "Please confirm the ticket ID and try again."
            )
        else:
            answer = format_ticket_status(ticket_id, ticket)

        return {
            "answer": answer,
            "trace": ["ticket_status_response: answer generated"],
        }

    def route(state: GraphState) -> Category:
        return state["category"]

    builder = StateGraph(GraphState)
    builder.add_node("load_ticket_database", load_ticket_database)
    builder.add_node("classify_ticket", classify_ticket)
    builder.add_node("billing_support", billing_support)
    builder.add_node("technical_support", technical_support)
    builder.add_node("account_support", account_support)
    builder.add_node("lookup_ticket_status", lookup_ticket_status)
    builder.add_node("ticket_status_response", ticket_status_response)
    builder.add_node("general_support", general_support)
    builder.add_edge(START, "load_ticket_database")
    builder.add_edge("load_ticket_database", "classify_ticket")
    builder.add_conditional_edges(
        "classify_ticket",
        route,
        {
            "billing": "billing_support",
            "technical": "technical_support",
            "account": "account_support",
            "ticket_status": "lookup_ticket_status",
            "general": "general_support",
        },
    )
    builder.add_edge("billing_support", END)
    builder.add_edge("technical_support", END)
    builder.add_edge("account_support", END)
    builder.add_edge("lookup_ticket_status", "ticket_status_response")
    builder.add_edge("ticket_status_response", END)
    builder.add_edge("general_support", END)
    return builder.compile()
