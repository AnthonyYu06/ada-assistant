"""Long-term memory tools: remember, forget, recall, and usage patterns.

Facts live in a plain local JSONL file via ``ada.brain.knowledge`` (see its
docstring for the privacy stance); usage patterns are computed from the
local audit log. Nothing here touches the network.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from datetime import datetime
from typing import Any

from ..brain.knowledge import KnowledgeStore
from . import get_config
from .registry import ToolError, ToolResult, tool

log = logging.getLogger("ada.tools.memory")

_DISABLED_MSG = "Long-term memory is turned off in the config."

# -- shared store ------------------------------------------------------------

_store_lock = threading.Lock()
_instance: KnowledgeStore | None = None
_instance_path = None


def get_store() -> KnowledgeStore:
    """The shared KnowledgeStore for the active config.

    Exposed so the orchestrator can reuse the same store for auto-recall
    (``context_for``) without re-reading the file. Rebuilt if the configured
    path changes (e.g. across test configs).
    """
    global _instance, _instance_path
    cfg = get_config()
    path = cfg.memory_path
    with _store_lock:
        if _instance is None or _instance_path != path:
            _instance = KnowledgeStore(path, max_facts=cfg.memory.max_facts)
            _instance_path = path
        return _instance


def _store() -> KnowledgeStore:
    """get_store() plus the enabled check every memory tool needs."""
    _ensure_enabled()
    return get_store()


def _ensure_enabled() -> None:
    if not get_config().memory.enabled:
        raise ToolError(_DISABLED_MSG)


# -- helpers -----------------------------------------------------------------


def _relative_age(ts: float) -> str:
    """A compact relative age like '3 days ago' for a unix timestamp."""
    try:
        delta = max(0.0, time.time() - float(ts))
    except (TypeError, ValueError):
        return ""
    if delta < 60:
        return "just now"
    minutes = int(delta // 60)
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = int(delta // 3600)
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = int(delta // 86400)
    if days < 30:
        return f"{days} day{'s' if days != 1 else ''} ago"
    months = days // 30
    if months < 12:
        return f"{months} month{'s' if months != 1 else ''} ago"
    years = days // 365
    return f"{years} year{'s' if years != 1 else ''} ago"


def _hour_phrase(hour: int) -> str:
    """0..23 -> '12 am' / '9 am' / '12 pm' / '3 pm'."""
    suffix = "am" if hour < 12 else "pm"
    return f"{hour % 12 or 12} {suffix}"


# -- tools -------------------------------------------------------------------


@tool(
    name="remember",
    description=(
        "Save a lasting fact or preference about the user to long-term "
        "memory, so it can be recalled in future conversations. Use when "
        "the user says 'remember ...' or states something durable about "
        "themselves (their gym, their coffee order, a preference)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": (
                    "The fact to remember, as a short standalone statement, "
                    "e.g. 'gym is Crunch Fitness on 5th'."
                ),
            },
        },
        "required": ["fact"],
    },
    risk="safe",
    category="memory",
)
def remember(fact: str) -> str:
    if not fact or not fact.strip():
        raise ToolError("Tell me what to remember.")
    _store().add(fact)
    return "Remembered."


@tool(
    name="forget",
    description=(
        "Delete facts from long-term memory whose text matches a phrase. "
        "Use when the user asks to forget something previously remembered."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text identifying the fact(s) to delete, e.g. 'gym'.",
            },
        },
        "required": ["query"],
    },
    risk="confirm",
    category="memory",
    describe_call=lambda a: f"Forget everything matching “{a.get('query', '')}”",
)
def forget(query: str) -> str:
    if not query or not query.strip():
        raise ToolError("Tell me what to forget.")
    count = _store().remove(query)
    if count == 0:
        raise ToolError("I don't have anything matching that.")
    if count == 1:
        return "Forgot 1 thing."
    return f"Forgot {count} things."


@tool(
    name="recall",
    description=(
        "Look up what is stored in long-term memory about the user. With a "
        "query, returns the best-matching facts; with no query, the most "
        "recent facts. Use when the user asks what you remember."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional topic to search memory for; empty for recent facts.",
                "default": "",
            },
        },
    },
    risk="safe",
    category="memory",
)
def recall(query: str = "") -> str | ToolResult:
    store = _store()
    query = (query or "").strip()
    facts = store.search(query, k=5) if query else store.recent(k=10)
    if not facts:
        if query:
            return "I don't have anything about that."
        return "I don't have anything remembered yet."
    items = [
        {"text": f["text"], "sub": _relative_age(f.get("created", 0.0))}
        for f in facts
    ]
    return ToolResult(
        speech="Here's what I remember.",
        display={"kind": "list", "title": "Memory", "items": items},
    )


# -- usage patterns ----------------------------------------------------------

# Friendlier phrasings for the most common tools; anything unmapped falls
# back to "use <tool name>".
_TOOL_PHRASES = {
    "open_app": "open apps",
    "quit_app": "quit apps",
    "open_url": "open web pages",
    "open_path": "open files",
    "search_files": "find files",
    "search_web": "search the web",
    "fetch_url": "read web pages",
    "set_timer": "set timers",
    "cancel_timers": "cancel timers",
    "create_reminder": "set reminders",
    "create_note": "take notes",
    "set_volume": "adjust the volume",
    "mute_audio": "mute the sound",
    "list_windows": "check windows",
    "window_action": "arrange windows",
    "battery": "check the battery",
    "remember": "remember things",
    "recall": "recall memories",
}


def _tool_phrase(name: str) -> str:
    return _TOOL_PHRASES.get(name, f"use {name.replace('_', ' ')}")


def _local_hour(ts: Any) -> int | None:
    """The local hour-of-day of an ISO-8601 timestamp, or None."""
    try:
        return datetime.fromisoformat(str(ts)).astimezone().hour
    except (TypeError, ValueError):
        return None


@tool(
    name="usage_patterns",
    description=(
        "Summarize how the user uses Ada from the local activity log: the "
        "most-used tools, most-opened apps, and the hours of day they talk "
        "to Ada. Use when the user asks about their habits or how they use "
        "the assistant."
    ),
    input_schema={"type": "object", "properties": {}},
    risk="safe",
    category="memory",
)
def usage_patterns() -> str | ToolResult:
    _ensure_enabled()
    path = get_config().log_dir / "audit.jsonl"
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return "I don't have enough history yet."

    tool_counts: Counter[str] = Counter()
    app_counts: Counter[str] = Counter()
    hour_counts: Counter[int] = Counter()
    command_count = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if not isinstance(rec, dict):
            continue
        kind = rec.get("kind")
        if kind == "tool_call":
            name = rec.get("tool")
            if isinstance(name, str) and name:
                tool_counts[name] += 1
                if name == "open_app":
                    args = rec.get("args")
                    app = args.get("app_name") if isinstance(args, dict) else None
                    if isinstance(app, str) and app:
                        app_counts[app] += 1
        elif kind == "transcript":
            command_count += 1
        else:
            continue
        hour = _local_hour(rec.get("ts"))
        if hour is not None:
            hour_counts[hour] += 1

    if not tool_counts and command_count == 0:
        return "I don't have enough history yet."

    top_tools = tool_counts.most_common(5)
    top_app = app_counts.most_common(1)[0][0] if app_counts else None
    busiest = hour_counts.most_common(1)[0][0] if hour_counts else None

    # One-sentence insight.
    if top_tools:
        lead = " and ".join(_tool_phrase(name) for name, _ in top_tools[:2])
        speech = f"You mostly ask me to {lead}"
        if top_app:
            speech += f", especially {top_app}"
    else:
        speech = f"You've given me {command_count} commands"
    if busiest is not None:
        speech += f", usually around {_hour_phrase(busiest)}"
    speech += "."

    # Chart when the hour histogram has enough points; key-value otherwise.
    display: dict[str, Any]
    if len(hour_counts) >= 5:
        hours = sorted(hour_counts)
        display = {
            "kind": "chart",
            "chart": "bar",
            "title": "When you talk to Ada",
            "labels": [_hour_phrase(h) for h in hours],
            "values": [hour_counts[h] for h in hours],
        }
    else:
        pairs: list[list[str]] = [
            [name, str(count)] for name, count in top_tools
        ]
        if top_app:
            pairs.append(["Most-opened app", top_app])
        if busiest is not None:
            pairs.append(["Busiest hour", _hour_phrase(busiest)])
        if command_count:
            pairs.append(["Voice commands", str(command_count)])
        display = {"kind": "keyvalue", "title": "Usage patterns", "pairs": pairs}

    return ToolResult(speech=speech, display=display)
