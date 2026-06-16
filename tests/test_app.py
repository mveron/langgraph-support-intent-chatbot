from app import CHAT_HISTORY_HEIGHT, render_chat_history


class FakeContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeStreamlit:
    def __init__(self):
        self.containers: list[dict[str, object]] = []
        self.chat_roles: list[str] = []
        self.writes: list[str] = []

    def container(self, **kwargs):
        self.containers.append(kwargs)
        return FakeContext()

    def chat_message(self, role: str):
        self.chat_roles.append(role)
        return FakeContext()

    def write(self, content: str):
        self.writes.append(content)


def test_render_chat_history_uses_fixed_height_scroll_container():
    fake_st = FakeStreamlit()

    render_chat_history(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "How can I help?"},
        ],
        streamlit_module=fake_st,
    )

    assert fake_st.containers == [
        {"height": CHAT_HISTORY_HEIGHT, "border": True}
    ]
    assert fake_st.chat_roles == ["user", "assistant"]
    assert fake_st.writes == ["Hello", "How can I help?"]
