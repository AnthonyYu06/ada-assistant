"""Shared brain machinery, independent of the LLM provider.

`BaseBrain` owns the parts that must behave identically no matter which model
answers: building the system prompt, and — most importantly — executing a
tool call safely (permission policy → optional confirmation → timeout →
audit). The Anthropic and Ollama brains subclass this and implement only the
provider-specific streaming loop.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from typing import Any

from ..config import Config
from ..safety.base import Confirmer, PermissionPolicy
from ..tools import registry
from ..tools.registry import ToolError, ToolResult
from ..ui import events as ui_events
from .prompts import build_system_prompt

log = logging.getLogger("ada.brain")

# Hard wall-clock limit for a single tool handler.
_TOOL_TIMEOUT_S = 20.0

REFUSAL_TEXT = "I can't help with that request."
EXHAUSTED_TEXT = "I stopped — that took more steps than I'm allowed."


class BaseBrain:
    """Provider-agnostic base: system prompt + safe tool execution + audit."""

    def __init__(
        self,
        cfg: Config,
        policy: PermissionPolicy,
        confirmer: Confirmer,
        audit: Any,
    ) -> None:
        """``audit`` is an ada.safety.audit.AuditLog (needs .record(kind, **fields))."""
        self._cfg = cfg
        self._policy = policy
        self._confirmer = confirmer
        self._audit = audit
        self._system = build_system_prompt(cfg)
        # Optional long-term memory (ada.brain.knowledge.KnowledgeStore),
        # wired by Assistant.build() when cfg.memory.enabled. Providers use it
        # to prefix auto-recalled facts onto the user message.
        self.knowledge: Any = None

    # -- tool execution ------------------------------------------------------
    def _execute_tool(
        self,
        name: str,
        args: dict[str, Any],
        cancel: threading.Event | None = None,
    ) -> tuple[str, bool]:
        """Run one tool call. Returns (result_text, is_error).

        Every call — allowed, denied, confirmed, failed, or timed out — is
        recorded in the audit log. ``cancel``, if set while the handler runs,
        aborts the wait promptly so a barge-in isn't blocked for up to the full
        tool timeout by one slow tool.
        """
        args = dict(args) if args else {}
        start = time.monotonic()
        allowed = False
        confirmed: bool | None = None
        ok = False
        error: str | None = None
        summary = name
        try:
            spec = registry.get(name)
            if spec is None:
                error = f"Unknown tool: {name}"
                return error, True
            summary = spec.summary(args)
            ui_events.publish(
                "tool", {"name": name, "summary": summary, "state": "running"}
            )

            decision = self._policy.evaluate(spec, args)
            if not decision.allowed:
                error = f"Denied by permission policy: {decision.reason}"
                return error, True
            allowed = True

            if decision.needs_confirmation:
                confirmed = self._confirmer.confirm(
                    title=f"{self._cfg.assistant.name} wants to act",
                    message=spec.summary(args),
                )
                if not confirmed:
                    error = "The user declined the confirmation dialog."
                    return error, True

            # Run the handler on a worker thread so a hung tool can't wedge
            # the whole assistant. Do NOT wait for a timed-out thread.
            pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"tool-{name}"
            )
            try:
                future = pool.submit(spec.handler, **args)
                try:
                    # Poll in short slices so a barge-in (cancel) is honoured
                    # promptly, while still enforcing the overall tool timeout.
                    deadline = time.monotonic() + _TOOL_TIMEOUT_S
                    while True:
                        try:
                            result = future.result(timeout=0.25)
                            break
                        except concurrent.futures.TimeoutError:
                            if cancel is not None and cancel.is_set():
                                future.cancel()
                                error = "The user cancelled this action."
                                log.info("Tool %s cancelled by barge-in.", name)
                                return error, True
                            if time.monotonic() >= deadline:
                                future.cancel()
                                error = f"The {name} tool timed out."
                                log.warning(
                                    "Tool %s exceeded %.0fs.", name, _TOOL_TIMEOUT_S
                                )
                                return error, True
                except ToolError as exc:
                    error = str(exc)
                    log.info("Tool %s reported: %s", name, error)
                    return error, True
                except Exception:
                    log.exception("Tool %s failed unexpectedly.", name)
                    error = "The tool failed unexpectedly."
                    return error, True
            finally:
                pool.shutdown(wait=False, cancel_futures=True)

            ok = True
            if isinstance(result, ToolResult):
                if result.display:
                    ui_events.publish("display", result.display)
                return result.speech, False
            return str(result), False
        finally:
            duration_ms = round((time.monotonic() - start) * 1000.0, 1)
            ui_events.publish(
                "tool",
                {
                    "name": name,
                    "summary": summary,
                    "state": "ok" if ok else "error",
                },
            )
            self._audit.record(
                "tool_call",
                tool=name,
                args=args,
                allowed=allowed,
                confirmed=confirmed,
                ok=ok,
                error=error,
                duration_ms=duration_ms,
            )
