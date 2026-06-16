NODES = (
    "START",
    "classify",
    "answer_technical",
    "answer_general",
    "END",
)

EDGES = (
    '"START" -> "classify"',
    '"classify" -> "answer_technical" [label=" technical"]',
    '"classify" -> "answer_general" [label=" general"]',
    '"answer_technical" -> "END"',
    '"answer_general" -> "END"',
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
