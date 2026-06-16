def render_dot(executed: list[str]) -> str:
    from visualization import graph_dot

    return graph_dot(executed)


def test_graph_dot_contains_both_conditional_routes():
    dot = render_dot([])

    assert '"classify" -> "answer_technical" [label=" technical"]' in dot
    assert '"classify" -> "answer_general" [label=" general"]' in dot


def test_graph_dot_highlights_executed_classifier_node():
    dot = render_dot(["classify"])

    assert '"classify" [style=filled, fillcolor="#22c55e"]' in dot


def test_graph_dot_marks_start_and_end_as_unexecuted_by_default():
    dot = render_dot([])

    assert '"START" [style=filled, fillcolor="#dbeafe"]' in dot
    assert '"END" [style=filled, fillcolor="#dbeafe"]' in dot
