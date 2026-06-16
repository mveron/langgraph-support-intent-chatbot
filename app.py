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
    st.title("Technical Support Classifier Chatbot")
    st.caption(
        "Chat with a support bot that classifies each message and routes it to "
        "billing, technical, account, ticket status, or general support."
    )

    if "executed" not in st.session_state:
        st.session_state.executed = []
    if "result" not in st.session_state:
        st.session_state.result = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Support Triage Graph")
        graph_slot = st.empty()
        graph_slot.graphviz_chart(graph_dot(st.session_state.executed))
        st.write(
            "Green nodes ran for the latest message. Blue nodes were not part "
            "of the selected support route. The dashed edge represents the next "
            "user turn re-entering the graph with conversation history."
        )

    with right:
        st.subheader("Chat")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        prompt = st.chat_input(
            "Ask about support, e.g. What is the status of TCK-1002?"
        )

        if prompt:
            conversation_history = list(st.session_state.messages)
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.executed = []
            st.session_state.result = None
            graph_slot.graphviz_chart(graph_dot(st.session_state.executed))

            with st.chat_message("user"):
                st.write(prompt)

            try:
                graph = build_graph(OllamaTextModel())
                with st.status(
                    "Running support chatbot graph...", expanded=True
                ) as status:
                    for event in stream_graph(
                        graph,
                        prompt,
                        conversation_history=conversation_history,
                    ):
                        st.session_state.executed.append(event.node)
                        graph_slot.graphviz_chart(graph_dot(st.session_state.executed))
                        st.write(f"Executed node: `{event.node}`")
                        st.write(f"Update: `{event.update}`")
                        st.session_state.result = dict(event.state)
                    status.update(
                        label="Support route complete",
                        state="complete",
                        expanded=False,
                    )

                answer = st.session_state.result.get(
                    "answer", "No support response generated."
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                with st.chat_message("assistant"):
                    st.write(answer)
            except OllamaUnavailableError as exc:
                error = str(exc)
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Error: {error}"}
                )
                st.error(error)

    if st.session_state.result:
        result = st.session_state.result
        st.subheader("Latest Triage Result")

        metric_a, metric_b = st.columns(2)
        metric_a.metric("Category", result.get("category", "no category"))
        metric_b.metric("Executed Nodes", len(st.session_state.executed))

        route = " -> ".join(st.session_state.executed) or "no execution"
        st.write(f"**Route:** {route}")
        if result.get("ticket_id"):
            st.write(f"**Ticket:** {result['ticket_id']}")

        with st.expander("Final State"):
            st.json(result)

    st.markdown(
        """
        ### Concepts

        **State:** shared data that moves between nodes, such as the support
        message, conversation history, mock ticket database, category, answer,
        and trace.

        **Node:** a graph function that reads state and returns updates.

        **Conditional edge:** a decision that chooses the next support route
        from part of the state, in this case the category. Ticket-status
        messages go through a lookup node before the response node.

        **Conversation loop:** each user turn starts a new graph run, but the
        previous chat messages are passed into `conversation_history`, so the
        graph can resolve follow-ups such as "what about that ticket?"

        **Stream:** step-by-step execution events that show the path as it runs.
        """
    )


if __name__ == "__main__":
    main()
