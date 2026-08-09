"""Short-term conversational memory.

Holds the Anthropic Messages API message list for the current conversation.
Assistant turns are stored exactly as returned by the SDK (the
``response.content`` block list), so tool_use blocks survive round trips.

The current local time is injected as a bracketed prefix on every user
message — this keeps the system prompt byte-stable (and therefore cacheable)
while still letting the model know what time it is.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

log = logging.getLogger("ada.brain.memory")


def _block_type(block: Any) -> str | None:
    """The content-block type, whether the block is a dict or an SDK object."""
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def _has_block(message: dict[str, Any], block_kind: str) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(_block_type(block) == block_kind for block in content)


class ConversationMemory:
    """Bounded, time-expiring message history for the Messages API."""

    def __init__(self, max_messages: int = 24, reset_after_min: int = 10) -> None:
        self.max_messages = max_messages
        self.reset_after_min = reset_after_min
        self.messages: list[dict[str, Any]] = []
        self._last_add: float | None = None  # time.monotonic() of the last add

    # -- adding turns -----------------------------------------------------
    def add_user(self, text: str) -> None:
        """Append a user message, prefixed with the current local time."""
        stamp = datetime.now().strftime("%A %Y-%m-%d %H:%M")
        self.messages.append({"role": "user", "content": f"[{stamp}] {text}"})
        self._mark()

    def add_assistant_content(self, content: Any) -> None:
        """Append an assistant turn (the SDK response.content list, as-is)."""
        self.messages.append({"role": "assistant", "content": content})
        self._mark()

    def add_tool_results(self, results: list[dict[str, Any]]) -> None:
        """Append one user message carrying all tool_result blocks for a turn."""
        self.messages.append({"role": "user", "content": results})
        self._mark()

    def _mark(self) -> None:
        self._last_add = time.monotonic()
        self.trim()

    # -- lifecycle ---------------------------------------------------------
    def maybe_reset(self) -> None:
        """Forget the conversation if it has gone stale."""
        if self._last_add is None:
            return
        if time.monotonic() - self._last_add > self.reset_after_min * 60:
            log.info(
                "Conversation idle for over %d min — starting fresh.",
                self.reset_after_min,
            )
            self.reset()

    def reset(self) -> None:
        """Clear everything."""
        self.messages.clear()
        self._last_add = None

    # -- trimming ------------------------------------------------------------
    def trim(self) -> None:
        """Bound the history without breaking tool_use / tool_result pairs.

        Messages are dropped from the front until at most ``max_messages``
        remain, then further dropped until the first message is a plain user
        message (the API requires messages[0] to be a user turn, and a leading
        bare tool_result message would orphan its tool_use).

        The trailing in-flight tool exchange — an assistant(tool_use) message
        immediately followed by its user(tool_result) carrier — is a protected
        tail that is NEVER trimmed. Dropping it would break the current tool
        loop or (at small history sizes) empty the list entirely. Both the
        length-based front-drop and the validity cascade stop before eating
        into that tail.
        """
        msgs = self.messages
        protected = self._protected_tail_len()
        # Length-based front-drop, but never into the protected tail.
        while len(msgs) > self.max_messages and len(msgs) > protected:
            msgs.pop(0)
        # Validity cascade, but never empty the list or eat the protected tail.
        while len(msgs) > protected and not self._valid_first(msgs[0]):
            msgs.pop(0)

    def _protected_tail_len(self) -> int:
        """Number of trailing messages forming the in-flight tool exchange.

        Returns 2 when the last message is a user tool_result carrier answered
        by an assistant(tool_use) message immediately before it, else 0. A lone
        or orphaned tool_result (no matching preceding tool_use) is NOT
        protected — it is cleaned up by the validity cascade.
        """
        msgs = self.messages
        if len(msgs) < 2:
            return 0
        last = msgs[-1]
        if last.get("role") != "user" or not _has_block(last, "tool_result"):
            return 0
        prev = msgs[-2]
        if prev.get("role") == "assistant" and _has_block(prev, "tool_use"):
            return 2
        return 0

    @staticmethod
    def _valid_first(message: dict[str, Any]) -> bool:
        if message.get("role") != "user":
            return False
        return not _has_block(message, "tool_result")
