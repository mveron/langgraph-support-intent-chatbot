NODES = (
    "START",
    "load_ticket_database",
    "classify_ticket",
    "billing_support",
    "technical_support",
    "account_support",
    "lookup_ticket_status",
    "ticket_status_response",
    "general_support",
    "END",
)

EDGES = (
    '"START" -> "classify_ticket"',
    '"classify_ticket" -> "billing_support" [label=" billing"]',
    '"classify_ticket" -> "technical_support" [label=" technical"]',
    '"classify_ticket" -> "account_support" [label=" account"]',
    '"classify_ticket" -> "load_ticket_database" [label=" ticket_status"]',
    '"classify_ticket" -> "general_support" [label=" general"]',
    '"billing_support" -> "END"',
    '"technical_support" -> "END"',
    '"account_support" -> "END"',
    '"load_ticket_database" -> "lookup_ticket_status"',
    '"lookup_ticket_status" -> "ticket_status_response"',
    '"ticket_status_response" -> "END"',
    '"general_support" -> "END"',
    '"END" -> "START" [style=dashed, label=" next user turn"]',
)

EXECUTED_COLOR = "#22c55e"
PENDING_COLOR = "#dbeafe"


def graph_dot(executed: list[str]) -> str:
    active = set(executed)
    lines = [
        "digraph langgraph_demo {",
        "  rankdir=LR;",
        "  node [shape=box, fontname=Helvetica];",
    ]

    for node in NODES:
        fillcolor = EXECUTED_COLOR if node in active else PENDING_COLOR
        lines.append(f'  "{node}" [style=filled, fillcolor="{fillcolor}"];')

    for edge in EDGES:
        lines.append(f"  {edge};")

    lines.append("}")
    return "\n".join(lines)
