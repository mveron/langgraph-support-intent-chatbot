from functools import lru_cache
from pathlib import Path
import re


DEFAULT_TICKET_DB_PATH = Path(__file__).parent / "data" / "mock_tickets.txt"
TicketDatabase = dict[str, dict[str, str]]


def parse_ticket_database(raw_text: str) -> TicketDatabase:
    tickets: TicketDatabase = {}

    for line in raw_text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        parts = [part.strip() for part in line.split("|")]
        ticket_id = parts[0].upper()
        fields: dict[str, str] = {}

        for part in parts[1:]:
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            fields[key.strip().lower()] = value.strip()

        if ticket_id:
            tickets[ticket_id] = fields

    return tickets


@lru_cache(maxsize=4)
def load_ticket_database(path: Path = DEFAULT_TICKET_DB_PATH) -> TicketDatabase:
    return parse_ticket_database(path.read_text(encoding="utf-8"))


def next_ticket_id(tickets: TicketDatabase) -> str:
    highest = 1000
    for ticket_id in tickets:
        match = re.fullmatch(r"TCK-(\d{4})", ticket_id)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"TCK-{highest + 1:04d}"


def build_mock_ticket_record(
    category: str,
    message: str,
    queue: str | None = None,
    priority: str = "normal",
    reason: str | None = None,
) -> dict[str, str]:
    owners = {
        "billing": "Billing",
        "technical": "Technical Support",
        "account": "Account Support",
        "general": "Support Triage",
    }
    summary = (reason or message).strip() or "Support request"
    owner = queue or owners.get(category, "Support Triage")
    return {
        "customer": "Demo User",
        "status": "Open",
        "summary": summary,
        "reason": summary,
        "owner": owner,
        "queue": owner,
        "priority": priority,
        "category": category,
        "updated": "2026-06-16",
    }


def extract_ticket_id(text: str) -> str | None:
    prefixed = re.search(r"\bTCK[-\s]?(\d{4})\b", text, flags=re.IGNORECASE)
    if prefixed:
        return f"TCK-{prefixed.group(1)}"

    plain = re.search(r"\bticket\s+#?(\d{4})\b", text, flags=re.IGNORECASE)
    if plain:
        return f"TCK-{plain.group(1)}"

    return None


def format_ticket_status(ticket_id: str, ticket: dict[str, str]) -> str:
    status = ticket.get("status", "Unknown")
    summary = ticket.get("summary", "No summary available")
    owner = ticket.get("owner", "Unassigned")
    queue = ticket.get("queue", owner)
    priority = ticket.get("priority", "normal")
    updated = ticket.get("updated", "No update date")
    customer = ticket.get("customer", "Unknown customer")
    return (
        f"{ticket_id} for {customer} is {status}. "
        f"Summary: {summary}. Queue: {queue}. Priority: {priority}. "
        f"Owner: {owner}. Last update: {updated}."
    )
