"""Productivity tools: Reminders, Notes, and in-process timers."""

from __future__ import annotations

import html
import itertools
import logging
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import datetime

from .applescript import applescript_date, applescript_quote, run_applescript
from .registry import ToolError, ToolResult, tool

log = logging.getLogger("ada.tools.productivity")

# -- timer plumbing ---------------------------------------------------------

# Each entry maps a label key to (unique-id, Timer). The id lets a firing
# timer deregister ONLY itself, so an old timer firing concurrently with a
# same-label replacement can't remove the new one.
_timers: dict[str, tuple[int, threading.Timer]] = {}
_timers_lock = threading.Lock()
_timer_seq = itertools.count(1)
_announcer: Callable[[str], None] | None = None


def set_announcer(fn: Callable[[str], None]) -> None:
    """Wire the function used to speak timer announcements (set by main.py)."""
    global _announcer
    _announcer = fn


def _natural_duration(seconds: int) -> str:
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours} hour" + ("s" if hours != 1 else ""))
    if minutes:
        parts.append(f"{minutes} minute" + ("s" if minutes != 1 else ""))
    if secs and not hours:
        parts.append(f"{secs} second" + ("s" if secs != 1 else ""))
    return " and ".join(parts) if parts else f"{seconds} seconds"


def _announce_phrase(seconds: int, label: str) -> str:
    if label:
        return f"Your {label} timer is done."
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours == 0 and secs == 0:
        return f"Your {minutes}-minute timer is done."
    if hours == 0 and minutes == 0:
        return f"Your {secs}-second timer is done."
    if minutes == 0 and secs == 0:
        return f"Your {hours}-hour timer is done."
    return f"Your {_natural_duration(seconds)} timer is done."


def _fire_timer(key: str, tid: int, message: str) -> None:
    """Timer callback — must never let an exception escape the thread."""
    with _timers_lock:
        entry = _timers.get(key)
        if entry is not None and entry[0] == tid:
            del _timers[key]
    try:
        script = f'display notification {applescript_quote(message)} with title "Ada timer"'
        subprocess.run(
            ["/usr/bin/osascript", "-e", script],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:  # noqa: BLE001
        log.exception("Timer notification failed")
    try:
        subprocess.run(
            ["/usr/bin/afplay", "/System/Library/Sounds/Glass.aiff"],
            capture_output=True, timeout=15,
        )
    except Exception:  # noqa: BLE001
        log.exception("Timer sound failed")
    announcer = _announcer
    if announcer is not None:
        try:
            announcer(message)
        except Exception:  # noqa: BLE001
            log.exception("Timer announcer failed")


# -- tools ------------------------------------------------------------------


@tool(
    name="create_reminder",
    description=(
        "Create a reminder in the macOS Reminders app, optionally with a due "
        "date and time. Use when the user asks to be reminded about something "
        "at a specific time or to add something to their reminders."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "What to be reminded about.",
            },
            "due_datetime": {
                "type": "string",
                "description": "Optional due date and time, ISO format like '2026-07-13T15:42'.",
            },
        },
        "required": ["title"],
    },
    risk="safe",
    category="productivity",
)
def create_reminder(title: str, due_datetime: str | None = None) -> str:
    if not title or not title.strip():
        raise ToolError("Tell me what the reminder should say.")
    title = title.strip()
    quoted = applescript_quote(title)
    if due_datetime:
        date_snippet = applescript_date(due_datetime)
        script = (
            f"{date_snippet}\n"
            'tell application "Reminders"\n'
            f"\tmake new reminder with properties {{name:{quoted}, remind me date:theDate}}\n"
            "end tell"
        )
        run_applescript(script, timeout=15.0)
        when = datetime.fromisoformat(due_datetime.strip())
        spoken = when.strftime("%B %d at %I:%M %p").replace(" 0", " ").lstrip("0")
        return f"Reminder created: {title}, for {spoken}."
    script = (
        'tell application "Reminders"\n'
        f"\tmake new reminder with properties {{name:{quoted}}}\n"
        "end tell"
    )
    run_applescript(script, timeout=15.0)
    return f"Reminder created: {title}."


@tool(
    name="create_note",
    description=(
        "Create a new note in the macOS Notes app with a title and body text. "
        "Use when the user asks to jot something down, save a note, or "
        "remember a piece of text."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "The note's title."},
            "body": {"type": "string", "description": "The note's body text."},
        },
        "required": ["title", "body"],
    },
    risk="safe",
    category="productivity",
)
def create_note(title: str, body: str) -> str:
    if not title or not title.strip():
        raise ToolError("Tell me what to call the note.")
    title = title.strip()
    # Notes bodies are HTML: escape the text, then turn newlines into <br>.
    body_html = html.escape(body).replace("\r\n", "\n").replace("\n", "<br>")
    script = (
        'tell application "Notes"\n'
        f"\tmake new note with properties {{name:{applescript_quote(title)}, "
        f"body:{applescript_quote(body_html)}}}\n"
        "end tell"
    )
    run_applescript(script, timeout=15.0)
    return f"Note created: {title}."


@tool(
    name="set_timer",
    description=(
        "Start a countdown timer. When it finishes, the user gets a "
        "notification, a chime, and a spoken announcement. Use for 'set a "
        "timer for N minutes', tea timers, pomodoros, and similar."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "duration_seconds": {
                "type": "integer",
                "description": "Timer length in seconds (1 to 86400).",
            },
            "label": {
                "type": "string",
                "description": "Optional short label, e.g. 'tea' or 'laundry'.",
                "default": "",
            },
        },
        "required": ["duration_seconds"],
    },
    risk="safe",
    category="productivity",
)
def set_timer(duration_seconds: int, label: str = "") -> ToolResult:
    try:
        seconds = int(duration_seconds)
    except (TypeError, ValueError):
        raise ToolError("Timer duration must be a number of seconds.") from None
    if not 1 <= seconds <= 86400:
        raise ToolError("Timer duration must be between 1 second and 24 hours.")
    label = (label or "").strip()
    tid = next(_timer_seq)
    key = label or f"timer-{tid}"
    message = _announce_phrase(seconds, label)

    timer = threading.Timer(seconds, _fire_timer, args=(key, tid, message))
    timer.daemon = True
    with _timers_lock:
        old = _timers.pop(key, None)
        _timers[key] = (tid, timer)
    if old is not None:
        old[1].cancel()
    timer.start()
    return ToolResult(
        speech=f"Timer set for {_natural_duration(seconds)}.",
        display={
            "kind": "timer",
            # Stable per timer, so HUD updates replace this card, not stack.
            "id": f"timer-{tid}",
            "label": label or "Timer",
            "ends_at": time.time() + seconds,
            "duration_s": seconds,
        },
    )


@tool(
    name="cancel_timers",
    description="Cancel all running timers. Use when the user asks to stop or cancel a timer.",
    input_schema={"type": "object", "properties": {}},
    risk="safe",
    category="productivity",
)
def cancel_timers() -> str:
    with _timers_lock:
        cancelled = [timer for _tid, timer in _timers.values()]
        _timers.clear()
    for timer in cancelled:
        timer.cancel()
    count = len(cancelled)
    if count == 0:
        return "There are no active timers."
    if count == 1:
        return "Cancelled 1 timer."
    return f"Cancelled {count} timers."
