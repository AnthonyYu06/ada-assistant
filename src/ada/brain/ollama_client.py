"""A local, API-free brain backed by Ollama.

Talks to a locally-running Ollama server (`http://localhost:11434` by default)
over its native `/api/chat` endpoint — no cloud, no API key, no per-token cost.
Streams the reply sentence-by-sentence for low-latency TTS and runs the same
bounded, permission-checked tool loop as the cloud brain.

Tool calling depends on the chosen model: Ollama only exposes tools for models
whose template supports them. If the model can't, Ada notices, says so once, and
carries on in chat-only mode instead of failing.

Thinking models (gemma4, qwen3, deepseek-r1...) are told not to reason
(`"think": false` by default) — silent reasoning adds seconds before the first
spoken word. Servers/models that reject the parameter get the same treatment
as tools: retry once without it, then omit it for good.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from typing import Any, Callable

import httpx

from ..config import Config
from ..safety.base import Confirmer, PermissionPolicy
from ..tools import registry
from .base import EXHAUSTED_TEXT, BaseBrain
from .sentences import SentenceSplitter

log = logging.getLogger("ada.brain.ollama")


class OllamaError(RuntimeError):
    """The Ollama server returned an error we can't recover from this turn."""


class OllamaUnavailable(OllamaError):
    """The Ollama server could not be reached (not running / wrong host)."""


class _ToolsUnsupported(Exception):
    """The selected model rejects the `tools` parameter — retry without it."""


class _ThinkUnsupported(Exception):
    """The server/model rejects the `think` parameter — retry without it."""


def _is_think_error(detail: str) -> bool:
    """Does this server error mean the `think` parameter was rejected?

    Ollama answers "<model> does not support thinking" when the model lacks
    the capability; servers predating the parameter reject the field itself
    with 'json: unknown field "think"'. Matched on those shapes only, so an
    unrelated error that merely mentions a model named "think..." can't trip
    the fallback.
    """
    lower = detail.lower()
    return "does not support think" in lower or (
        "unknown field" in lower and "think" in lower
    )


def _extract_error(data: bytes) -> str:
    try:
        obj = json.loads(data)
        if isinstance(obj, dict) and obj.get("error"):
            return str(obj["error"])
    except (json.JSONDecodeError, ValueError):
        pass
    return data[:300].decode("utf-8", errors="replace")


class OllamaMemory:
    """Bounded, time-expiring history in Ollama/OpenAI message format.

    Messages are plain dicts: {"role": "user"|"assistant"|"tool", ...}. The
    system message is NOT stored here (the brain prepends a constant one each
    request). trim() never splits an assistant tool-call message from the tool
    results that answer it, and keeps the first message a plain user turn.
    """

    def __init__(self, max_messages: int = 24, reset_after_min: int = 10) -> None:
        self.max_messages = max_messages
        self.reset_after_min = reset_after_min
        self.messages: list[dict[str, Any]] = []
        self._last_add: float | None = None

    def add_user(self, text: str, context: str = "") -> None:
        """Append a user turn. ``context`` is auto-recalled long-term memory —
        it rides in the same bracketed prefix as the timestamp so the system
        prompt (and the cacheable prefix) stays static."""
        stamp = datetime.now().strftime("%A %Y-%m-%d %H:%M")
        prefix = f"[{stamp}] " if not context else f"[{stamp} | {context}] "
        self.messages.append({"role": "user", "content": f"{prefix}{text}"})
        self._mark()

    def add_assistant(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        self._mark()

    def add_tool(self, tool_name: str, content: str) -> None:
        self.messages.append(
            {"role": "tool", "tool_name": tool_name, "content": content}
        )
        self._mark()

    def maybe_reset(self, now: float | None = None) -> None:
        if self._last_add is None:
            return
        import time

        current = time.monotonic() if now is None else now
        if current - self._last_add > self.reset_after_min * 60:
            log.info("Conversation idle over %d min — starting fresh.", self.reset_after_min)
            self.reset()

    def reset(self) -> None:
        self.messages.clear()
        self._last_add = None

    def _mark(self) -> None:
        import time

        self._last_add = time.monotonic()
        self.trim()

    def trim(self) -> None:
        msgs = self.messages
        protected = self._protected_tail_len()
        while len(msgs) > self.max_messages and len(msgs) > protected:
            msgs.pop(0)
        # First message must be a plain user turn.
        while msgs and msgs[0].get("role") != "user" and len(msgs) > protected:
            msgs.pop(0)

    def _protected_tail_len(self) -> int:
        """How many trailing messages form the current in-flight tool exchange."""
        msgs = self.messages
        n = len(msgs)
        if n == 0:
            return 0
        i = n - 1
        if msgs[i].get("role") == "tool":
            while i >= 0 and msgs[i].get("role") == "tool":
                i -= 1
            if i >= 0 and msgs[i].get("role") == "assistant" and msgs[i].get("tool_calls"):
                return n - i  # assistant(tool_calls) + its tool results
        return min(1, n)


class OllamaBrain(BaseBrain):
    """Conversation orchestrator backed by a local Ollama model."""

    def __init__(
        self,
        cfg: Config,
        policy: PermissionPolicy,
        confirmer: Confirmer,
        audit: Any,
    ) -> None:
        super().__init__(cfg, policy, confirmer, audit)
        oc = cfg.llm.ollama
        self._model = oc.model
        self._host = oc.host.rstrip("/")
        self._temperature = oc.temperature
        self._num_ctx = oc.num_ctx
        self._keep_alive = oc.keep_alive
        self._think = oc.think
        self._system_msg = {"role": "system", "content": self._system}
        self._client = httpx.Client(
            base_url=self._host,
            timeout=httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0),
        )
        self.memory = OllamaMemory(
            max_messages=cfg.llm.history_max_messages,
            reset_after_min=cfg.llm.reset_history_after_min,
        )
        self._tools = self._ollama_tools()
        self._tools_supported = True
        self._tools_warned = False
        self._think_supported = True  # cleared if the server rejects `think`

    # -- lifecycle -----------------------------------------------------------
    def warmup(self) -> None:
        """Check the server + model, preload the model, prime the prompt cache.

        The preload request mirrors a real turn exactly — same system message,
        tools, options, keep_alive and think — so the first turn reuses the
        warmed KV prefix (~1.6k prompt tokens of system + tool schemas) instead
        of re-evaluating it, and the model is loaded with the same num_ctx
        (loading with different options would force a full reload on the first
        real request). num_predict=1 caps the generation cost. Never raises.
        """
        try:
            resp = self._client.get("/api/tags", timeout=5.0)
        except httpx.HTTPError as exc:
            log.warning(
                "Could not reach Ollama at %s (%s). Start it with `ollama serve` "
                "and pull a model with `ollama pull %s`.",
                self._host, exc, self._model,
            )
            return
        if resp.status_code != 200:
            log.warning("Ollama responded HTTP %s at %s", resp.status_code, self._host)
            return

        names = [m.get("name", "") for m in resp.json().get("models", [])]
        if not self._model_present(names):
            log.warning(
                "Ollama model %r is not pulled (have: %s). Run: ollama pull %s",
                self._model, ", ".join(names) or "none", self._model,
            )
            return
        log.info(
            "Ollama ready: model=%s at %s, %d tool(s) registered.",
            self._model, self._host, len(self._tools),
        )
        for _ in range(2):  # the request + at most one retry without `think`
            body = self._chat_body(
                [self._system_msg],
                stream=False,
                use_tools=self._tools_supported and bool(self._tools),
            )
            body["options"]["num_predict"] = 1  # warmup only: one token is enough
            try:
                resp = self._client.post("/api/chat", json=body, timeout=120.0)
            except httpx.HTTPError as exc:
                log.debug("Model preload failed (non-fatal): %s", exc)
                return
            if resp.status_code == 200:
                return
            detail = _extract_error(resp.content)
            if self._think_supported and _is_think_error(detail):
                self._disable_think(detail)
                continue
            log.debug(
                "Model preload failed (non-fatal): HTTP %s: %s",
                resp.status_code, detail,
            )
            return

    def _model_present(self, names: list[str]) -> bool:
        if self._model in names:
            return True
        # Allow "gemma3" to match a pulled "gemma3:4b", and vice-versa.
        base = self._model.split(":")[0]
        return any(n == self._model or n.split(":")[0] == base for n in names)

    def close(self) -> None:
        self._client.close()

    # -- main entry point ----------------------------------------------------
    def handle(
        self,
        user_text: str,
        on_sentence: Callable[[str], None],
        cancel: threading.Event | None = None,
    ) -> str:
        """Answer one utterance via the local model, streaming + running tools."""
        self.memory.maybe_reset()
        context = ""
        if self.knowledge is not None:
            try:
                context = self.knowledge.context_for(
                    user_text, self._cfg.memory.inject_top_k
                )
            except Exception:  # noqa: BLE001 - recall must never break a turn
                log.exception("Long-term memory recall failed")
        self.memory.add_user(user_text, context)

        splitter = SentenceSplitter()
        texts: list[str] = []

        def emit(sentence: str) -> None:
            if sentence:
                on_sentence(sentence)

        def joined(partial: str = "") -> str:
            parts = [t for t in texts if t.strip()]
            if partial.strip():
                parts.append(partial)
            return " ".join(p.strip() for p in parts)

        for _ in range(self._cfg.llm.max_tool_iterations):
            if cancel is not None and cancel.is_set():
                return joined()

            try:
                text, tool_calls, cancelled = self._stream_turn(splitter, emit, cancel)
            except OllamaUnavailable as exc:
                log.warning("Ollama unreachable: %s", exc)
                msg = "I can't reach the local model. Make sure Ollama is running."
                emit(msg)
                return msg
            except OllamaError as exc:
                log.error("Ollama error: %s", exc)
                msg = "The local model returned an error."
                emit(msg)
                return msg

            if cancelled:
                return joined(text)

            texts.append(text)
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": text}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            self.memory.add_assistant(assistant_msg)

            if not tool_calls:
                emit(splitter.flush())
                return joined()

            # Run each tool and feed the results back as tool-role messages.
            emit(splitter.flush())
            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name") or ""
                args = self._coerce_args(fn.get("arguments"))
                result_text, _is_error = self._execute_tool(name, args, cancel)
                self.memory.add_tool(name, result_text)
            if cancel is not None and cancel.is_set():
                return joined()

        emit(EXHAUSTED_TEXT)
        return EXHAUSTED_TEXT

    # -- streaming -----------------------------------------------------------
    def _stream_turn(
        self,
        splitter: SentenceSplitter,
        emit: Callable[[str], None],
        cancel: threading.Event | None,
    ) -> tuple[str, list[dict[str, Any]], bool]:
        try:
            return self._stream_with_tools_fallback(splitter, emit, cancel)
        except _ThinkUnsupported as exc:
            # The server rejected `think`: retry this request once without it
            # and omit it from every later request (including warmup's).
            self._disable_think(str(exc))
            return self._stream_with_tools_fallback(splitter, emit, cancel)

    def _stream_with_tools_fallback(
        self,
        splitter: SentenceSplitter,
        emit: Callable[[str], None],
        cancel: threading.Event | None,
    ) -> tuple[str, list[dict[str, Any]], bool]:
        use_tools = self._tools_supported and bool(self._tools)
        try:
            return self._do_stream(splitter, emit, cancel, use_tools)
        except _ToolsUnsupported as exc:
            self._tools_supported = False
            if not self._tools_warned:
                self._tools_warned = True
                log.warning("Model %s does not support tools; chatting only (%s).", self._model, exc)
                emit("Heads up — this local model can chat but can't run Mac actions.")
            return self._do_stream(splitter, emit, cancel, use_tools=False)

    def _disable_think(self, detail: str) -> None:
        """Drop the `think` parameter for the rest of the process. Logs once —
        both call sites guard on _think_supported, so this can't run twice."""
        self._think_supported = False
        log.warning(
            "Model %s rejected the think parameter; retrying without it and "
            "omitting it from now on (%s).", self._model, detail,
        )

    def _chat_body(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
        use_tools: bool,
    ) -> dict[str, Any]:
        """One /api/chat body shape shared by real turns and warmup — the two
        must match exactly for the warmed KV prompt prefix to actually hit."""
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": self._temperature, "num_ctx": self._num_ctx},
            "keep_alive": self._keep_alive,
        }
        if self._think_supported:
            # False stops thinking models silently reasoning for seconds
            # before the first spoken word; harmless on non-thinking models.
            body["think"] = self._think
        if use_tools:
            body["tools"] = self._tools
        return body

    def _do_stream(
        self,
        splitter: SentenceSplitter,
        emit: Callable[[str], None],
        cancel: threading.Event | None,
        use_tools: bool,
    ) -> tuple[str, list[dict[str, Any]], bool]:
        body = self._chat_body(
            [self._system_msg, *self.memory.messages], stream=True, use_tools=use_tools
        )

        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        thinking_deltas = 0
        thinking_chars = 0
        cancelled = False
        try:
            with self._client.stream("POST", "/api/chat", json=body) as resp:
                if resp.status_code != 200:
                    self._raise_stream_error(_extract_error(resp.read()), resp.status_code)
                for line in resp.iter_lines():
                    if cancel is not None and cancel.is_set():
                        cancelled = True
                        break
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("error"):
                        self._raise_stream_error(str(obj["error"]))
                    message = obj.get("message") or {}
                    thinking = message.get("thinking") or ""
                    if thinking:
                        # Reasoning deltas from a thinking model — never fed
                        # to the sentence splitter or stored in memory: they
                        # are not speech and would be read aloud verbatim.
                        thinking_deltas += 1
                        thinking_chars += len(thinking)
                    delta = message.get("content") or ""
                    if delta:
                        content_parts.append(delta)
                        for sentence in splitter.feed(delta):
                            emit(sentence)
                    for call in message.get("tool_calls") or []:
                        tool_calls.append(call)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OllamaUnavailable(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(str(exc)) from exc
        if thinking_deltas:
            log.debug(
                "Ignored %d thinking delta(s) (%d chars) from %s.",
                thinking_deltas, thinking_chars, self._model,
            )
        return "".join(content_parts), tool_calls, cancelled

    def _raise_stream_error(self, detail: str, status: int | None = None) -> None:
        """Map a server error to the matching exception (tools/think retries)."""
        if "does not support tools" in detail.lower():
            raise _ToolsUnsupported(detail)
        if self._think_supported and _is_think_error(detail):
            raise _ThinkUnsupported(detail)
        raise OllamaError(detail if status is None else f"HTTP {status}: {detail}")

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _coerce_args(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _ollama_tools() -> list[dict[str, Any]]:
        """Convert the tool registry to Ollama/OpenAI function-tool schemas."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.input_schema,
                },
            }
            for spec in registry.all_tools()
        ]
