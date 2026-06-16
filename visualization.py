NODES = (
    "START",
    "classify_ticket",
    "billing_support",
    "technical_support",
    "account_support",
    "general_support",
    "END",
)

EDGES = (
    '"START" -> "classify_ticket"',
    '"classify_ticket" -> "billing_support" [label=" billing"]',
    '"classify_ticket" -> "technical_support" [label=" technical"]',
    '"classify_ticket" -> "account_support" [label=" account"]',
    '"classify_ticket" -> "general_support" [label=" general"]',
    '"billing_support" -> "END"',
    '"technical_support" -> "END"',
    '"account_support" -> "END"',
    '"general_support" -> "END"',
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
