"""The Anthropic-powered brain.

Streams a response from the Messages API, feeding text deltas through a
SentenceSplitter so TTS can start speaking early, and runs a bounded tool
loop: every tool call is checked against the permission policy, optionally
confirmed by the user, executed with a timeout, audited, and its result
returned to the model.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

import anthropic

from ..config import Config
from ..safety.base import Confirmer, PermissionPolicy
from ..tools import registry
from .base import EXHAUSTED_TEXT, REFUSAL_TEXT, BaseBrain
from .memory import ConversationMemory
from .sentences import SentenceSplitter

log = logging.getLogger("ada.brain")

_REFUSAL_TEXT = REFUSAL_TEXT
_EXHAUSTED_TEXT = EXHAUSTED_TEXT


class AnthropicBrain(BaseBrain):
    """Conversation orchestrator backed by the Anthropic Messages API."""

    def __init__(
        self,
        cfg: Config,
        policy: PermissionPolicy,
        confirmer: Confirmer,
        audit: Any,
    ) -> None:
        """``audit`` is an ada.safety.audit.AuditLog (needs .record(kind, **fields))."""
        if not cfg.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to your environment "
                "or a .env file next to config.yaml (or set llm.provider: ollama "
                "to run a local model instead)."
            )
        super().__init__(cfg, policy, confirmer, audit)
        self._client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        self.memory = ConversationMemory(
            max_messages=cfg.llm.history_max_messages,
            reset_after_min=cfg.llm.reset_history_after_min,
        )

    # -- lifecycle -----------------------------------------------------------
    def warmup(self) -> None:
        """Verify the brain is ready (client and prompt are built in __init__)."""
        tools = registry.anthropic_tools()
        if not tools:
            log.warning(
                "No tools registered — %s can chat but cannot act on the Mac.",
                self._cfg.assistant.name,
            )
        log.info(
            "Brain ready: anthropic model=%s effort=%s, %d tool(s) registered.",
            self._cfg.llm.anthropic.model,
            self._cfg.llm.anthropic.effort,
            len(tools),
        )

    # -- main entry point ------------------------------------------------------
    def handle(
        self,
        user_text: str,
        on_sentence: Callable[[str], None],
        cancel: threading.Event | None = None,
    ) -> str:
        """Answer one user utterance.

        Streams the model's reply, calling ``on_sentence`` with each speakable
        chunk as it completes, executing tool calls between model turns.
        Returns the full response text (for logging / console display).
        """
        self.memory.maybe_reset()
        self.memory.add_user(user_text)

        splitter = SentenceSplitter()
        texts: list[str] = []  # completed per-iteration response texts

        def emit(sentence: str) -> None:
            if sentence:
                on_sentence(sentence)

        def text_so_far(partial: str = "") -> str:
            parts = [t for t in texts if t.strip()]
            if partial.strip():
                parts.append(partial)
            return " ".join(p.strip() for p in parts)

        for _ in range(self._cfg.llm.max_tool_iterations):
            if cancel is not None and cancel.is_set():
                return text_so_far()

            current: list[str] = []
            cancelled = False
            try:
                with self._client.messages.stream(
                    model=self._cfg.llm.anthropic.model,
                    max_tokens=self._cfg.llm.anthropic.max_tokens,
                    system=[
                        {
                            "type": "text",
                            "text": self._system,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    tools=registry.anthropic_tools(),
                    messages=self._messages_with_cache_breakpoint(),
                    output_config={"effort": self._cfg.llm.anthropic.effort},
                ) as stream:
                    for event in stream:
                        if cancel is not None and cancel.is_set():
                            cancelled = True
                            break  # the context manager closes the stream
                        if (
                            event.type == "content_block_delta"
                            and event.delta.type == "text_delta"
                        ):
                            current.append(event.delta.text)
                            for sentence in splitter.feed(event.delta.text):
                                emit(sentence)
                    if cancelled:
                        return text_so_far("".join(current))
                    final = stream.get_final_message()
            except anthropic.AuthenticationError:
                log.error("Anthropic API key was rejected.")
                sentence = "My API key was rejected. Please check my configuration."
                emit(sentence)
                return sentence
            except anthropic.RateLimitError:
                log.warning("Rate limited by the Anthropic API.")
                sentence = "I'm being rate limited, give me a moment."
                emit(sentence)
                return sentence
            except anthropic.APIConnectionError:
                log.warning("Could not reach the Anthropic API.")
                sentence = "I can't reach the Anthropic API right now."
                emit(sentence)
                return sentence
            except anthropic.APIStatusError as exc:
                log.error(
                    "Anthropic API error %s: %s", exc.status_code, exc.message
                )
                sentence = "The API returned an error."
                emit(sentence)
                return sentence

            texts.append("".join(current))
            self.memory.add_assistant_content(final.content)

            if final.stop_reason == "refusal":
                emit(_REFUSAL_TEXT)
                return _REFUSAL_TEXT

            if final.stop_reason != "tool_use":
                # A turn that stops for any other reason (most often
                # "max_tokens" truncating a tool_use block mid-input) can STILL
                # carry tool_use blocks. The assistant turn was just stored; if
                # we return now, those blocks have no matching tool_result and
                # every later request 400s. Answer each one with a synthetic
                # error result. Do NOT strip the blocks — a tool_use-only
                # truncated turn would be left with empty content, also a 400.
                orphaned = [
                    b
                    for b in final.content
                    if getattr(b, "type", None) == "tool_use"
                ]
                if orphaned:
                    self.memory.add_tool_results(
                        [
                            {
                                "type": "tool_result",
                                "tool_use_id": b.id,
                                "content": "Response was cut off before this tool could run.",
                                "is_error": True,
                            }
                            for b in orphaned
                        ]
                    )
                emit(splitter.flush())
                return text_so_far()

            # Model wants tools: speak any pending text, run every tool_use
            # block in order, and hand all results back in ONE user message.
            emit(splitter.flush())
            results: list[dict[str, Any]] = []
            for block in final.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                result_text, is_error = self._execute_tool(
                    getattr(block, "name", ""),
                    dict(getattr(block, "input", {}) or {}),
                    cancel,
                )
                item: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                }
                if is_error:
                    item["is_error"] = True
                results.append(item)
            self.memory.add_tool_results(results)
            # A barge-in during tool execution must not start another model
            # turn — record the results (so the history stays valid) and stop.
            if cancel is not None and cancel.is_set():
                return text_so_far()

        emit(_EXHAUSTED_TEXT)
        return _EXHAUSTED_TEXT

    # -- prompt caching ----------------------------------------------------------
    def _messages_with_cache_breakpoint(self) -> list[dict[str, Any]]:
        """A shallow copy of the memory messages with a cache breakpoint at the tail.

        The system block already carries an ephemeral ``cache_control``, but the
        system prefix (~2.3-2.7k tokens with tools) is below claude-opus-4-8's
        4096-token minimum, so that breakpoint alone silently never caches.
        Placing a second breakpoint on the last content block of the last
        message caches the growing conversation prefix across turns.

        ``self.memory.messages`` is never mutated in place; a non-dict content
        block is never annotated.
        """
        messages = self.memory.messages
        if not messages:
            return messages
        out = list(messages)
        last = dict(out[-1])
        content = last.get("content")
        if isinstance(content, str):
            last["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        elif isinstance(content, list) and content and isinstance(content[-1], dict):
            new_content = list(content)
            tail = dict(new_content[-1])
            tail["cache_control"] = {"type": "ephemeral"}
            new_content[-1] = tail
            last["content"] = new_content
        out[-1] = last
        return out


# Backwards-compatible alias (the Anthropic brain used to be the only one).
Brain = AnthropicBrain
