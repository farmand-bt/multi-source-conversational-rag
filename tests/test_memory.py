"""Unit tests for ConversationMemory.

The LLM call inside rewrite_query is mocked so tests run offline.
"""

from unittest.mock import MagicMock

import pytest

from rag.memory.conversation import ConversationMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory_with_mock_llm(rewrite_response: str = "standalone question") -> ConversationMemory:
    mem = ConversationMemory()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=rewrite_response)
    mem._llm = mock_llm
    return mem


# ---------------------------------------------------------------------------
# add_turn / get_history / clear
# ---------------------------------------------------------------------------

class TestConversationMemoryState:
    def test_empty_history_on_init(self):
        mem = ConversationMemory()
        assert mem.get_history() == []

    def test_add_turn_appends_both_roles(self):
        mem = ConversationMemory()
        mem.add_turn("What is Python?", "Python is a programming language.")
        history = mem.get_history()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "What is Python?"}
        assert history[1] == {"role": "assistant", "content": "Python is a programming language."}

    def test_multiple_turns_preserved_in_order(self):
        mem = ConversationMemory()
        mem.add_turn("Q1", "A1")
        mem.add_turn("Q2", "A2")
        history = mem.get_history()
        assert [m["content"] for m in history] == ["Q1", "A1", "Q2", "A2"]

    def test_clear_resets_history(self):
        mem = ConversationMemory()
        mem.add_turn("hello", "hi")
        mem.clear()
        assert mem.get_history() == []

    def test_turns_capped_at_max_history_turns(self, monkeypatch):
        monkeypatch.setattr("rag.memory.conversation.MAX_HISTORY_TURNS", 2)
        mem = ConversationMemory()
        mem.add_turn("Q1", "A1")
        mem.add_turn("Q2", "A2")
        mem.add_turn("Q3", "A3")  # should evict Q1/A1
        history = mem.get_history()
        contents = [m["content"] for m in history]
        assert "Q1" not in contents
        assert "Q3" in contents


# ---------------------------------------------------------------------------
# rewrite_query
# ---------------------------------------------------------------------------

class TestRewriteQuery:
    def test_no_history_returns_original_query(self):
        mem = ConversationMemory()
        result = mem.rewrite_query("What is it?", history=[])
        assert result == "What is it?"

    def test_no_history_no_turns_returns_original(self):
        mem = ConversationMemory()
        result = mem.rewrite_query("Tell me more.")
        assert result == "Tell me more."

    def test_with_history_calls_llm(self):
        mem = _make_memory_with_mock_llm("What is machine learning?")
        history = [
            {"role": "user", "content": "Tell me about AI."},
            {"role": "assistant", "content": "AI stands for..."},
        ]
        result = mem.rewrite_query("What about that?", history=history)
        assert result == "What is machine learning?"
        mem._llm.invoke.assert_called_once()

    def test_uses_passed_history_over_internal_turns(self):
        mem = _make_memory_with_mock_llm("rewritten")
        mem.add_turn("internal Q", "internal A")
        external_history = [{"role": "user", "content": "external"}, {"role": "assistant", "content": "answer"}]
        mem.rewrite_query("follow-up", history=external_history)
        # Verify the prompt sent to the LLM contains the external history text
        call_args = mem._llm.invoke.call_args[0][0]  # list of messages
        prompt_text = call_args[0].content
        assert "external" in prompt_text
        assert "internal Q" not in prompt_text

    def test_empty_llm_response_falls_back_to_original(self):
        mem = _make_memory_with_mock_llm("")
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        result = mem.rewrite_query("follow-up?", history=history)
        assert result == "follow-up?"

    def test_llm_response_is_stripped(self):
        mem = _make_memory_with_mock_llm("  trimmed question  ")
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        result = mem.rewrite_query("follow-up?", history=history)
        assert result == "trimmed question"
