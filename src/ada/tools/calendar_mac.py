"""Calendar tools: read and create events in the macOS Calendar app.

Performance note: asking Calendar for "every event" is pathologically slow on
large calendars. calendar_events therefore filters with a whose-clause on
start-date bounds, caps the result at 15 events, and does everything in a
single osascript invocation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from .applescript import applescript_date, applescript_quote, run_applescript
from .registry import ToolError, ToolResult, tool

log = logging.getLogger("ada.tools.calendar")

_SEP = "|||"
_MAX_EVENTS = 15
_RANGES = ("today", "tomorrow", "week")
_MAX_DURATION_MIN = 24 * 60


def _date_snippet(var: str, dt: datetime) -> str:
    """AppleScript snippet defining `var` as a date (month-overflow safe)."""
    return "\n".join(
        [
            f"set {var} to current date",
            f"set day of {var} to 1",
            f"set year of {var} to {dt.year}",
            f"set month of {var} to {dt.month}",
            f"set day of {var} to {dt.day}",
            f"set time of {var} to ({dt.hour} * hours + {dt.minute} * minutes + {dt.second})",
        ]
    )


def _range_bounds(range: str, now: datetime) -> tuple[datetime, datetime, str]:
    """Return (start, end, spoken label) for a named range."""
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if range == "today":
        return midnight, midnight + timedelta(days=1), "today"
    if range == "tomorrow":
        start = midnight + timedelta(days=1)
        return start, start + timedelta(days=1), "tomorrow"
    return midnight, midnight + timedelta(days=7), "this week"


def _events_script(start: datetime, end: datetime) -> str:
    """One-shot script: whose-filtered events, capped, one line per event.

    Each line is sortKey|||title|||endKey|||location|||calendar where the
    keys encode Y/M/D/minutes as a single sortable integer.
    """
    key = (
        "(year of {d}) * 100000000 + ((month of {d}) as integer) * 1000000 "
        "+ (day of {d}) * 10000 + (time of {d}) div 60"
    )
    return "\n".join(
        [
            _date_snippet("theStart", start),
            _date_snippet("theEnd", end),
            f'set sep to "{_SEP}"',
            "set outLines to {}",
            "set collected to 0",
            'tell application "Calendar"',
            "\trepeat with c in calendars",
            "\t\tset calName to name of c",
            "\t\tset evs to {}",
            "\t\ttry",
            "\t\t\tset evs to (every event of c whose start date is greater than "
            "or equal to theStart and start date is less than theEnd)",
            "\t\tend try",
            "\t\trepeat with e in evs",
            f"\t\t\tif collected is greater than or equal to {_MAX_EVENTS} then exit repeat",
            "\t\t\tset sd to start date of e",
            "\t\t\tset ed to sd",
            "\t\t\ttry",
            "\t\t\t\tset ed to end date of e",
            "\t\t\tend try",
            '\t\t\tset loc to ""',
            "\t\t\ttry",
            "\t\t\t\tset loc to location of e",
            "\t\t\tend try",
            '\t\t\tif loc is missing value then set loc to ""',
            f"\t\t\tset sKey to {key.format(d='sd')}",
            f"\t\t\tset eKey to {key.format(d='ed')}",
            "\t\t\tset end of outLines to (sKey as text) & sep & (summary of e) "
            "& sep & (eKey as text) & sep & loc & sep & calName",
            "\t\t\tset collected to collected + 1",
            "\t\tend repeat",
            f"\t\tif collected is greater than or equal to {_MAX_EVENTS} then exit repeat",
            "\tend repeat",
            "end tell",
            "set AppleScript's text item delimiters to linefeed",
            "return outLines as text",
        ]
    )


def _parse_key(raw: str) -> datetime | None:
    """Decode a Y*1e8 + M*1e6 + D*1e4 + minutes sort key to a datetime."""
    try:
        key = int(raw.strip())
        minutes = key % 10000
        day = (key // 10000) % 100
        month = (key // 1000000) % 100
        year = key // 100000000
        return datetime(year, month, day, minutes // 60, minutes % 60)
    except ValueError:
        return None


def _spoken_time(dt: datetime) -> str:
    hour = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    return f"{hour}:{dt.minute:02d} {ampm}"


@tool(
    name="calendar_events",
    description=(
        "List the user's Calendar events for today, tomorrow, or the next "
        "seven days. Use when the user asks what's on their calendar, their "
        "schedule, or their next meeting."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "range": {
                "type": "string",
                "enum": list(_RANGES),
                "description": "Which period to list: today, tomorrow, or the coming week.",
            }
        },
        "required": ["range"],
    },
    risk="safe",
    category="calendar",
)
def calendar_events(range: str) -> str | ToolResult:
    if range not in _RANGES:
        raise ToolError("The range must be today, tomorrow, or week.")
    now = datetime.now()
    start, end, label = _range_bounds(range, now)
    out = run_applescript(_events_script(start, end), timeout=15.0)

    parsed: list[tuple[datetime, dict]] = []
    for line in out.splitlines():
        parts = line.split(_SEP)
        if len(parts) != 5:
            continue
        start_dt = _parse_key(parts[0])
        if start_dt is None:
            continue
        title = parts[1].strip() or "Untitled event"
        event: dict = {"title": title, "start": f"{start_dt.strftime('%a')} {_spoken_time(start_dt)}"}
        end_dt = _parse_key(parts[2])
        if end_dt is not None and end_dt != start_dt:
            event["end"] = f"{end_dt.strftime('%a')} {_spoken_time(end_dt)}"
        if parts[3].strip():
            event["location"] = parts[3].strip()
        if parts[4].strip():
            event["calendar"] = parts[4].strip()
        parsed.append((start_dt, event))

    if not parsed:
        return f"You have no events {label}."

    parsed.sort(key=lambda pair: pair[0])
    count = len(parsed)
    if count >= _MAX_EVENTS:
        speech = f"You have {_MAX_EVENTS} or more events {label}."
    else:
        noun = "event" if count == 1 else "events"
        speech = f"You have {count} {noun} {label}."
    upcoming = next(((dt, ev) for dt, ev in parsed if dt >= now), None)
    if upcoming is not None:
        dt, ev = upcoming
        if dt.date() == now.date():
            when = f"at {_spoken_time(dt)}"
        else:
            when = f"{dt.strftime('%A')} at {_spoken_time(dt)}"
        speech += f" Next is {ev['title']} {when}."
    return ToolResult(
        speech=speech,
        display={
            "kind": "calendar",
            "title": f"Events {label}",
            "events": [ev for _dt, ev in parsed],
        },
    )


@tool(
    name="create_event",
    description=(
        "Create an event in the macOS Calendar app with a title, a start "
        "date and time, a duration in minutes, and an optional location. Use "
        "when the user asks to schedule, book, or put something on their "
        "calendar."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "The event title."},
            "start": {
                "type": "string",
                "description": "Start date and time, like '2026-07-14 15:00'.",
            },
            "duration_min": {
                "type": "integer",
                "description": "Length in minutes (default 60).",
                "default": 60,
            },
            "location": {
                "type": "string",
                "description": "Optional location for the event.",
            },
        },
        "required": ["title", "start"],
    },
    risk="confirm",
    category="calendar",
    describe_call=lambda a: (
        f"Create calendar event '{a['title']}' at {a['start']}"
        + (f" ({a['duration_min']} min)" if a.get("duration_min") else "")
    ),
)
def create_event(
    title: str,
    start: str,
    duration_min: int = 60,
    location: str | None = None,
) -> str:
    if not title or not title.strip():
        raise ToolError("Tell me what to call the event.")
    title = title.strip()
    try:
        when = datetime.fromisoformat(str(start).strip())
    except (TypeError, ValueError):
        raise ToolError(
            f"I couldn't understand the start time {start!r} — "
            "use the format 2026-07-14 15:00."
        ) from None
    try:
        minutes = int(duration_min)
    except (TypeError, ValueError):
        raise ToolError("The duration must be a number of minutes.") from None
    if not 1 <= minutes <= _MAX_DURATION_MIN:
        raise ToolError("The duration must be between 1 minute and 24 hours.")

    properties = f"summary:{applescript_quote(title)}, start date:theDate, end date:endDate"
    if location and location.strip():
        properties += f", location:{applescript_quote(location.strip())}"
    script = (
        f"{applescript_date(start)}\n"
        f"set endDate to theDate + ({minutes} * minutes)\n"
        'tell application "Calendar"\n'
        "\tset targetCal to first calendar whose writable is true\n"
        "\ttell targetCal\n"
        f"\t\tmake new event with properties {{{properties}}}\n"
        "\tend tell\n"
        "end tell"
    )
    run_applescript(script, timeout=15.0)
    spoken = when.strftime("%B %d at %I:%M %p").replace(" 0", " ").lstrip("0")
    return f"Event created: {title}, {spoken}."
