import streamlit as st

from graph import build_graph
from llm import OllamaTextModel, OllamaUnavailableError
from runner import stream_graph
from visualization import graph_dot


def main() -> None:
    st.set_page_config(
        page_title="LangGraph + Ollama",
        page_icon="🧭",
        layout="wide",
    )
    st.title("Interactive LangGraph + Ollama Demo")
    st.caption(
        "See how a question moves through classification, conditional routing, "
        "and answer generation."
    )

    if "executed" not in st.session_state:
        st.session_state.executed = []
    if "result" not in st.session_state:
        st.session_state.result = None

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Execution Graph")
        graph_slot = st.empty()
        graph_slot.graphviz_chart(graph_dot(st.session_state.executed))
        st.write(
            "Green nodes have already run for the current question. "
            "Blue nodes have not been part of the route yet."
        )

    with right:
        st.subheader("Question")
        with st.form("question_form"):
            question = st.text_input(
                "Question",
                placeholder="Example: Explain what a Python decorator is",
            )
            submitted = st.form_submit_button("Run Graph")

        if submitted:
            if not question.strip():
                st.warning("Enter a question before running the graph.")
            else:
                st.session_state.executed = []
                st.session_state.result = None
                graph_slot.graphviz_chart(graph_dot(st.session_state.executed))

                try:
                    graph = build_graph(OllamaTextModel())
                    with st.status("Running graph...", expanded=True) as status:
                        for event in stream_graph(graph, question.strip()):
                            st.session_state.executed.append(event.node)
                            graph_slot.graphviz_chart(
                                graph_dot(st.session_state.executed)
                            )
                            st.write(f"Executed node: `{event.node}`")
                            st.write(f"Update: `{event.update}`")
                            st.session_state.result = dict(event.state)
                        status.update(
                            label="Execution complete",
                            state="complete",
                            expanded=False,
                        )
                except OllamaUnavailableError as exc:
                    st.error(str(exc))

    if st.session_state.result:
        result = st.session_state.result
        st.subheader("Result")
        st.write(result.get("answer", "No answer generated."))

        metric_a, metric_b = st.columns(2)
        metric_a.metric("Category", result.get("category", "no category"))
        metric_b.metric("Executed Nodes", len(st.session_state.executed))

        route = " -> ".join(st.session_state.executed) or "no execution"
        st.write(f"**Route:** {route}")

        with st.expander("Final State"):
            st.json(result)

    st.markdown(
        """
        ### Concepts

        **State:** shared data that moves between nodes, such as the question,
        category, answer, and trace.

        **Node:** a graph function that reads state and returns updates.

        **Conditional edge:** a decision that chooses the next node from part
        of the state, in this case the category.

        **Stream:** step-by-step execution events that show the path as it runs.
        """
    )


if __name__ == "__main__":
    main()
