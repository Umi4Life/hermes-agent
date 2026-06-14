from agent.chat_completion_helpers import _maybe_emit_incompetent_fallback_warning


class FakeAgent:
    def __init__(self):
        self.warnings = []
        self.notices = []

    def _emit_warning(self, text):
        self.warnings.append(text)

    def _emit_notice(self, notice):
        self.notices.append(notice)


def test_warning_emitted_for_tsukishiro_qwen_fallback():
    agent = FakeAgent()

    _maybe_emit_incompetent_fallback_warning(
        agent,
        old_model="composer-2.5",
        fb_model="tsukishiro-qwen3-5-9b",
        fb_provider="custom:tsukishiro-litellm",
    )

    assert len(agent.warnings) == 1
    warning = agent.warnings[0]
    assert "MODEL FALLBACK WARNING" in warning
    assert "tsukishiro-qwen3-5-9b" in warning
    assert "EXTREMELY INCOMPETENT" in warning
    assert "Chatbot tier only" in warning
    assert len(agent.notices) == 1
    assert "incompetent tier" in agent.notices[0].text
    assert agent.notices[0].level == "warn"
    assert agent.notices[0].kind == "sticky"


def test_warning_not_emitted_for_non_qwen_fallback():
    agent = FakeAgent()

    _maybe_emit_incompetent_fallback_warning(
        agent,
        old_model="gpt-5.5",
        fb_model="composer-2.5",
        fb_provider="cursor-sdk",
    )

    assert agent.warnings == []
    assert agent.notices == []
