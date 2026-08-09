"""Tests for ada.brain.memory.ConversationMemory: stamps, trimming, expiry."""

from __future__ import annotations

import re
import types
from typing import Any

import pytest

from ada.brain import memory as memory_mod
from ada.brain.memory import ConversationMemory


class _FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    """Replace the memory module's time binding with a controllable clock."""
    fake = _FakeClock()
    monkeypatch.setattr(memory_mod, "time", types.SimpleNamespace(monotonic=fake))
    return fake


def _tool_use(block_id: str) -> dict[str, Any]:
    return {"type": "tool_use", "id": block_id, "name": "open_app", "input": {}}


def _tool_result(block_id: str) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": block_id, "content": "ok"}


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _assert_valid_history(messages: list[dict[str, Any]]) -> None:
    """API invariants: first message is a plain user turn; every tool_result
    block is answered by a tool_use with the same id in the message before it."""
    if not messages:
        return
    first = messages[0]
    assert first["role"] == "user"
    if isinstance(first["content"], list):
        assert all(
            not (isinstance(b, dict) and b.get("type") == "tool_result")
            for b in first["content"]
        )
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                assert i > 0, "leading tool_result message"
                prev = messages[i - 1]["content"]
                assert isinstance(prev, list)
                use_ids = {
                    b.get("id")
                    for b in prev
                    if isinstance(b, dict) and b.get("type") == "tool_use"
                }
                assert block["tool_use_id"] in use_ids


# -- add_user timestamp prefix ------------------------------------------------

def test_add_user_prefixes_timestamp() -> None:
    mem = ConversationMemory()
    mem.add_user("what time is it?")
    assert len(mem.messages) == 1
    msg = mem.messages[0]
    assert msg["role"] == "user"
    # "[Monday 2026-07-13 15:42] what time is it?"
    assert re.fullmatch(
        r"\[[A-Z][a-z]+ \d{4}-\d{2}-\d{2} \d{2}:\d{2}\] what time is it\?",
        msg["content"],
    )


# -- trimming ------------------------------------------------------------------

def test_trim_keeps_history_bounded_and_first_user() -> None:
    mem = ConversationMemory(max_messages=6)
    for i in range(10):
        mem.add_user(f"question {i}")
        mem.add_assistant_content([_text(f"answer {i}")])
    assert len(mem.messages) <= 6
    assert mem.messages[0]["role"] == "user"
    _assert_valid_history(mem.messages)
    # The most recent turn is retained.
    assert mem.messages[-1]["content"][0]["text"] == "answer 9"


def test_tool_pair_survives_when_within_budget() -> None:
    mem = ConversationMemory(max_messages=10)
    mem.add_user("open safari")
    mem.add_assistant_content([_tool_use("t1")])
    mem.add_tool_results([_tool_result("t1")])
    mem.add_assistant_content([_text("Opened Safari.")])
    assert len(mem.messages) == 4
    _assert_valid_history(mem.messages)


def test_trim_never_separates_tool_use_from_tool_result() -> None:
    mem = ConversationMemory(max_messages=5)
    mem.add_user("q1")
    mem.add_assistant_content([_tool_use("t1")])
    mem.add_tool_results([_tool_result("t1")])
    mem.add_assistant_content([_text("done")])
    mem.add_user("q2")
    assert len(mem.messages) == 5
    # This add overflows the budget; the tool_use/tool_result pair at the
    # front must be dropped together, never leaving an orphaned tool_result.
    mem.add_assistant_content([_text("again")])
    assert len(mem.messages) <= 5
    _assert_valid_history(mem.messages)
    for msg in mem.messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                assert block.get("type") != "tool_result"


def test_inflight_tool_exchange_survives_small_history() -> None:
    # The trailing assistant(tool_use) + user(tool_result) pair is the
    # in-flight exchange of the current tool loop. trim() must never drop it,
    # nor empty the list, even when max_messages is smaller than the pair.
    mem = ConversationMemory(max_messages=2)
    mem.add_user("open safari")
    mem.add_assistant_content([_tool_use("t1")])
    mem.add_tool_results([_tool_result("t1")])
    # A naive front-drop/validity cascade would pop the whole exchange here.
    assert mem.messages != []
    assert len(mem.messages) == 2
    assert mem.messages[-2]["role"] == "assistant"
    assert memory_mod._has_block(mem.messages[-2], "tool_use")
    assert mem.messages[-1]["role"] == "user"
    assert memory_mod._has_block(mem.messages[-1], "tool_result")
    # The tool_result is still answered by the tool_use immediately before it.
    use_ids = {b["id"] for b in mem.messages[-2]["content"]}
    assert mem.messages[-1]["content"][0]["tool_use_id"] in use_ids


def test_first_message_always_user() -> None:
    mem = ConversationMemory(max_messages=8)
    # An assistant turn with no preceding user turn is invalid and dropped.
    mem.add_assistant_content([_text("hello?")])
    assert mem.messages == []
    # A bare tool_result carrier with no tool_use before it is also dropped.
    mem.add_tool_results([_tool_result("t9")])
    assert mem.messages == []
    mem.add_user("hi")
    assert mem.messages[0]["role"] == "user"


# -- maybe_reset ---------------------------------------------------------------

def test_maybe_reset_clears_after_idle_window(clock: _FakeClock) -> None:
    mem = ConversationMemory(reset_after_min=10)
    mem.add_user("hello")
    mem.add_assistant_content([_text("hi there")])

    clock.advance(9 * 60)
    mem.maybe_reset()
    assert len(mem.messages) == 2  # still fresh

    clock.advance(2 * 60)  # now 11 min since the last add
    mem.maybe_reset()
    assert mem.messages == []


def test_maybe_reset_noop_when_empty(clock: _FakeClock) -> None:
    mem = ConversationMemory(reset_after_min=10)
    mem.maybe_reset()  # must not raise with no prior activity
    assert mem.messages == []


def test_activity_restarts_idle_window(clock: _FakeClock) -> None:
    mem = ConversationMemory(reset_after_min=10)
    mem.add_user("one")
    clock.advance(9 * 60)
    mem.add_user("two")  # refreshes the idle clock
    clock.advance(9 * 60)
    mem.maybe_reset()
    assert len(mem.messages) == 2  # only 9 min since the last add


def test_reset_clears_everything(clock: _FakeClock) -> None:
    mem = ConversationMemory()
    mem.add_user("hello")
    mem.reset()
    assert mem.messages == []
    clock.advance(10_000)
    mem.maybe_reset()  # no-op after reset
    assert mem.messages == []
