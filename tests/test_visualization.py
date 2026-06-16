def render_dot(executed: list[str]) -> str:
    from visualization import graph_dot

    return graph_dot(executed)


def test_graph_dot_contains_both_conditional_routes():
    dot = render_dot([])

    assert '"START" -> "prepare_context"' in dot
    assert '"prepare_context" -> "classify_ticket"' in dot
    assert '"classify_ticket" -> "assess_ticket_need" [label=" billing"]' in dot
    assert '"classify_ticket" -> "assess_ticket_need" [label=" technical"]' in dot
    assert '"classify_ticket" -> "assess_ticket_need" [label=" account"]' in dot
    assert '"assess_ticket_need" -> "load_ticket_database" [label=" create_ticket"]' in dot
    assert '"load_ticket_database" -> "create_ticket" [label=" create"]' in dot
    assert '"create_ticket" -> "billing_support" [label=" billing"]' in dot
    assert '"assess_ticket_need" -> "billing_support" [label=" respond_only billing"]' in dot
    assert '"classify_ticket" -> "load_ticket_database" [label=" ticket_status"]' in dot
    assert '"load_ticket_database" -> "lookup_ticket_status" [label=" lookup"]' in dot
    assert '"classify_ticket" -> "general_support" [label=" general"]' in dot
    assert '"lookup_ticket_status" -> "ticket_status_response"' in dot
    assert '"ticket_status_response" -> "END"' in dot


def test_graph_dot_highlights_executed_classifier_node():
    dot = render_dot(["classify_ticket"])

    assert '"classify_ticket" [style=filled, fillcolor="#22c55e"]' in dot


def test_graph_dot_shows_dashed_conversation_loop_for_next_turn():
    dot = render_dot([])

    assert '"END" -> "START" [style=dashed, label=" next user turn"]' in dot


def test_graph_dot_marks_start_and_end_as_unexecuted_by_default():
    dot = render_dot([])

    assert '"START" [style=filled, fillcolor="#dbeafe"]' in dot
    assert '"END" [style=filled, fillcolor="#dbeafe"]' in dot
