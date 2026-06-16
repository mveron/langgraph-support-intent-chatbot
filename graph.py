import operator
import re
from typing import Annotated, Literal, Protocol

from typing_extensions import Required, TypedDict

from langgraph.graph import END, START, StateGraph

from ticket_db import (
    TicketDatabase,
    build_mock_ticket_record,
    extract_ticket_id,
    format_ticket_status,
    load_ticket_database as load_mock_ticket_database,
    next_ticket_id,
)

Category = Literal["billing", "technical", "account", "ticket_status", "general"]
TicketAction = Literal["create_ticket", "respond_only"]


class ConversationMessage(TypedDict):
    role: str
    content: str


class TextModel(Protocol):
    def generate(self, prompt: str) -> str: ...


class GraphState(TypedDict, total=False):
    message: Required[str]
    conversation_history: list[ConversationMessage]
    customer_context: str
    support_reason: str
    ticket_database: TicketDatabase
    ticket_action: TicketAction
    ticket_id: str
    ticket_record: dict[str, str]
    ticket_queue: str
    ticket_priority: str
    ticket_decision_reason: str
    category: Category
    answer: str
    trace: Annotated[list[str], operator.add]


class GraphUpdate(TypedDict, total=False):
    conversation_history: list[ConversationMessage]
    customer_context: str
    support_reason: str
    ticket_database: TicketDatabase
    ticket_action: TicketAction
    ticket_id: str
    ticket_record: dict[str, str]
    ticket_queue: str
    ticket_priority: str
    ticket_decision_reason: str
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


def last_user_message(history: list[ConversationMessage] | None) -> str | None:
    for message in reversed(history or []):
        if message.get("role") == "user" and message.get("content", "").strip():
            return message["content"].strip()
    return None


def summarize_support_reason(
    message: str, history: list[ConversationMessage] | None
) -> str:
    current = message.strip()
    previous = last_user_message(history)
    word_count = len(re.findall(r"\w+", current))
    if previous and word_count <= 4:
        return f"Follow-up: {current.rstrip('.')}. Previous issue: {previous}"
    return current or "Support request"


def assign_ticket_queue(category: Category, reason: str) -> str:
    normalized = reason.lower()
    if category == "billing":
        return "billing_operations"
    if category == "account":
        return "identity_access"
    if category == "technical":
        if any(keyword in normalized for keyword in ("api", "integration", "webhook")):
            return "technical_integrations"
        return "technical_support_tier_2"
    return "support_triage"


def infer_ticket_priority(category: Category, reason: str) -> str:
    normalized = reason.lower()
    high_keywords = (
        "charged twice",
        "cannot log in",
        "can't log in",
        "locked out",
        "crash",
        "crashes",
        "500",
        "down",
        "outage",
        "production",
    )
    if any(keyword in normalized for keyword in high_keywords):
        return "high"
    if category == "billing" and any(
        keyword in normalized for keyword in ("payment", "charge", "invoice")
    ):
        return "normal"
    return "normal"


def should_create_ticket(category: Category, message: str) -> bool:
    if category not in {"billing", "technical", "account"}:
        return False

    normalized = message.strip().lower()
    informational_prefixes = (
        "how do i",
        "how can i",
        "where can i",
        "what is",
        "can you explain",
        "help me understand",
    )
    return not normalized.startswith(informational_prefixes)


def ticket_context(state: GraphState) -> str:
    ticket_id = state.get("ticket_id")
    ticket = state.get("ticket_record")
    if not ticket_id or not ticket:
        return "No ticket created for this turn."
    return (
        f"Created ticket: {ticket_id}. Status: {ticket.get('status', 'Unknown')}. "
        f"Queue: {ticket.get('queue', 'Unassigned')}. "
        f"Priority: {ticket.get('priority', 'normal')}. "
        f"Summary: {ticket.get('summary', 'No summary available')}."
    )


def add_ticket_notice(state: GraphState, answer: str) -> str:
    ticket_id = state.get("ticket_id")
    ticket = state.get("ticket_record")
    if state.get("ticket_action") != "create_ticket" or not ticket_id or not ticket:
        return answer
    return (
        f"I created support ticket {ticket_id} in the "
        f"{ticket.get('queue', 'support_triage')} queue with "
        f"{ticket.get('priority', 'normal')} priority. {answer}"
    )


def build_graph(model: TextModel):
    def prepare_context(state: GraphState) -> GraphUpdate:
        history = state.get("conversation_history", [])
        return {
            "customer_context": format_conversation_history(history),
            "support_reason": summarize_support_reason(state["message"], history),
            "trace": ["prepare_context: reason captured"],
        }

    def load_ticket_database(state: GraphState) -> GraphUpdate:
        tickets = load_mock_ticket_database()
        return {
            "ticket_database": tickets,
            "trace": [f"load_ticket_database: loaded {len(tickets)} tickets"],
        }

    def assess_ticket_need(state: GraphState) -> GraphUpdate:
        action: TicketAction = (
            "create_ticket"
            if should_create_ticket(state["category"], state["message"])
            else "respond_only"
        )
        reason = state.get("support_reason", state["message"])
        if action == "create_ticket":
            queue = assign_ticket_queue(state["category"], reason)
            priority = infer_ticket_priority(state["category"], reason)
        else:
            queue = "none"
            priority = "none"
        decision_reason = (
            f"{state['category']} issue requires follow-up by {queue}."
            if action == "create_ticket"
            else "Self-service guidance can answer this request."
        )
        return {
            "ticket_action": action,
            "ticket_queue": queue,
            "ticket_priority": priority,
            "ticket_decision_reason": decision_reason,
            "trace": [f"assess_ticket_need: action={action}"],
        }

    def create_ticket(state: GraphState) -> GraphUpdate:
        ticket_database = dict(state.get("ticket_database", {}))
        ticket_id = next_ticket_id(ticket_database)
        ticket_record = build_mock_ticket_record(
            category=state["category"],
            message=state["message"],
            queue=state.get("ticket_queue", "support_triage"),
            priority=state.get("ticket_priority", "normal"),
            reason=state.get("support_reason", state["message"]),
        )
        ticket_database[ticket_id] = ticket_record
        return {
            "ticket_database": ticket_database,
            "ticket_id": ticket_id,
            "ticket_record": ticket_record,
            "trace": [f"create_ticket: ticket_id={ticket_id}"],
        }

    def classify_ticket(state: GraphState) -> GraphUpdate:
        history = state.get("customer_context") or format_conversation_history(
            state.get("conversation_history")
        )
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
            f"Support reason:\n{state.get('support_reason', state['message'])}\n"
            f"Current message: {state['message']}\n/no_think"
        )
        category = normalize_category(model.generate(prompt))
        return {
            "category": category,
            "trace": [f"classify_ticket: category={category}"],
        }

    def billing_support(state: GraphState) -> GraphUpdate:
        history = state.get("customer_context") or format_conversation_history(
            state.get("conversation_history")
        )
        prompt = (
            "You are a billing support specialist. Acknowledge the issue, ask "
            "for invoice or payment details if needed, and explain the next step.\n"
            f"Conversation history:\n{history}\n"
            f"Support reason: {state.get('support_reason', state['message'])}\n"
            f"Decision: {state.get('ticket_action', 'respond_only')}\n"
            f"Queue: {state.get('ticket_queue', 'none')}\n"
            f"Priority: {state.get('ticket_priority', 'none')}\n"
            f"Ticket context:\n{ticket_context(state)}\n"
            f"Current message: {state['message']}\n/no_think"
        )
        answer = add_ticket_notice(state, model.generate(prompt).strip())
        return {
            "answer": answer,
            "trace": ["billing_support: answer generated"],
        }

    def technical_support(state: GraphState) -> GraphUpdate:
        history = state.get("customer_context") or format_conversation_history(
            state.get("conversation_history")
        )
        prompt = (
            "You are a technical support specialist. Ask for the error message, "
            "environment, and reproduction steps when useful.\n"
            f"Conversation history:\n{history}\n"
            f"Support reason: {state.get('support_reason', state['message'])}\n"
            f"Decision: {state.get('ticket_action', 'respond_only')}\n"
            f"Queue: {state.get('ticket_queue', 'none')}\n"
            f"Priority: {state.get('ticket_priority', 'none')}\n"
            f"Ticket context:\n{ticket_context(state)}\n"
            f"Current message: {state['message']}\n/no_think"
        )
        answer = add_ticket_notice(state, model.generate(prompt).strip())
        return {
            "answer": answer,
            "trace": ["technical_support: answer generated"],
        }

    def account_support(state: GraphState) -> GraphUpdate:
        history = state.get("customer_context") or format_conversation_history(
            state.get("conversation_history")
        )
        prompt = (
            "You are an account support specialist. Help with login, password, "
            "profile, access, or verification issues without requesting secrets.\n"
            f"Conversation history:\n{history}\n"
            f"Support reason: {state.get('support_reason', state['message'])}\n"
            f"Decision: {state.get('ticket_action', 'respond_only')}\n"
            f"Queue: {state.get('ticket_queue', 'none')}\n"
            f"Priority: {state.get('ticket_priority', 'none')}\n"
            f"Ticket context:\n{ticket_context(state)}\n"
            f"Current message: {state['message']}\n/no_think"
        )
        answer = add_ticket_notice(state, model.generate(prompt).strip())
        return {
            "answer": answer,
            "trace": ["account_support: answer generated"],
        }

    def general_support(state: GraphState) -> GraphUpdate:
        history = state.get("customer_context") or format_conversation_history(
            state.get("conversation_history")
        )
        prompt = (
            "You are a support triage assistant. Give a helpful first response "
            "and ask one clarifying question if the issue is unclear.\n"
            f"Conversation history:\n{history}\n"
            f"Support reason: {state.get('support_reason', state['message'])}\n"
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

    def route_after_assessment(state: GraphState) -> str:
        if state["ticket_action"] == "create_ticket":
            return "create_ticket"
        return f"respond_only_{state['category']}"

    def route_after_database_load(state: GraphState) -> str:
        if state.get("category") == "ticket_status":
            return "lookup_ticket_status"
        return "create_ticket"

    builder = StateGraph(GraphState)
    builder.add_node("prepare_context", prepare_context)
    builder.add_node("load_ticket_database", load_ticket_database)
    builder.add_node("assess_ticket_need", assess_ticket_need)
    builder.add_node("create_ticket", create_ticket)
    builder.add_node("classify_ticket", classify_ticket)
    builder.add_node("billing_support", billing_support)
    builder.add_node("technical_support", technical_support)
    builder.add_node("account_support", account_support)
    builder.add_node("lookup_ticket_status", lookup_ticket_status)
    builder.add_node("ticket_status_response", ticket_status_response)
    builder.add_node("general_support", general_support)
    builder.add_edge(START, "prepare_context")
    builder.add_edge("prepare_context", "classify_ticket")
    builder.add_conditional_edges(
        "classify_ticket",
        route,
        {
            "technical": "assess_ticket_need",
            "account": "assess_ticket_need",
            "billing": "assess_ticket_need",
            "ticket_status": "load_ticket_database",
            "general": "general_support",
        },
    )
    builder.add_conditional_edges(
        "assess_ticket_need",
        route_after_assessment,
        {
            "create_ticket": "load_ticket_database",
            "respond_only_billing": "billing_support",
            "respond_only_technical": "technical_support",
            "respond_only_account": "account_support",
        },
    )
    builder.add_conditional_edges(
        "load_ticket_database",
        route_after_database_load,
        {
            "create_ticket": "create_ticket",
            "lookup_ticket_status": "lookup_ticket_status",
        },
    )
    builder.add_conditional_edges(
        "create_ticket",
        route,
        {
            "billing": "billing_support",
            "technical": "technical_support",
            "account": "account_support",
        },
    )
    builder.add_edge("billing_support", END)
    builder.add_edge("technical_support", END)
    builder.add_edge("account_support", END)
    builder.add_edge("lookup_ticket_status", "ticket_status_response")
    builder.add_edge("ticket_status_response", END)
    builder.add_edge("general_support", END)
    return builder.compile()
