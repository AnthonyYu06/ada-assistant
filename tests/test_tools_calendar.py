"""Tests for ada.tools.calendar_mac: event listing and creation.

AppleScript is mocked at the import site. Event rows use the module's
sortable key encoding (Y*1e8 + M*1e6 + D*1e4 + minutes-of-day).
"""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta

import pytest

from ada.tools import registry
from ada.tools.registry import ToolError, ToolResult

calendar_mac = importlib.import_module("ada.tools.calendar_mac")


def _key(dt: datetime) -> int:
    return (
        dt.year * 100000000
        + dt.month * 1000000
        + dt.day * 10000
        + dt.hour * 60
        + dt.minute
    )


def _row(
    title: str,
    start: datetime,
    end: datetime | None = None,
    location: str = "",
    calendar: str = "Work",
) -> str:
    end = end or start
    return f"{_key(start)}|||{title}|||{_key(end)}|||{location}|||{calendar}"


def _fake_applescript(
    monkeypatch: pytest.MonkeyPatch, responses: list[object]
) -> list[str]:
    calls: list[str] = []
    queue = list(responses)

    def fake(script: str, timeout: float = 10.0) -> str:
        calls.append(script)
        result = queue.pop(0) if queue else ""
        if isinstance(result, Exception):
            raise result
        return str(result)

    monkeypatch.setattr("ada.tools.calendar_mac.run_applescript", fake)
    return calls


# -- calendar_events ----------------------------------------------------------

def test_calendar_events_card_sorted_with_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now()
    earlier = now - timedelta(hours=2)
    later = now + timedelta(hours=2)
    # Deliberately out of order (per-calendar enumeration isn't sorted).
    rows = "\n".join(
        [
            _row("Standup", later, later + timedelta(hours=1), "Room 1"),
            _row("Breakfast", earlier, calendar="Home"),
        ]
    )
    calls = _fake_applescript(monkeypatch, [rows])
    result = calendar_mac.calendar_events("today")
    assert isinstance(result, ToolResult)
    assert result.speech.startswith("You have 2 events today.")
    assert "Next is Standup" in result.speech
    card = result.display
    assert card["kind"] == "calendar"
    assert [e["title"] for e in card["events"]] == ["Breakfast", "Standup"]
    standup = card["events"][1]
    assert standup["location"] == "Room 1"
    assert standup["calendar"] == "Work"
    assert "end" in standup  # differs from start
    assert "end" not in card["events"][0]  # same as start -> omitted
    # The script filters with a whose-clause and caps the enumeration.
    assert "whose start date is greater than or equal to theStart" in calls[0]
    assert "15" in calls[0]


def test_calendar_events_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_applescript(monkeypatch, [""])
    assert calendar_mac.calendar_events("tomorrow") == "You have no events tomorrow."


def test_calendar_events_week_label(monkeypatch: pytest.MonkeyPatch) -> None:
    start = datetime.now() + timedelta(days=2)
    _fake_applescript(monkeypatch, [_row("Dentist", start)])
    result = calendar_mac.calendar_events("week")
    assert result.speech.startswith("You have 1 event this week.")
    assert start.strftime("%A") in result.speech  # day name for non-today events


def test_calendar_events_at_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    base = datetime.now() + timedelta(hours=1)
    rows = "\n".join(
        _row(f"Event {i}", base + timedelta(minutes=i)) for i in range(15)
    )
    _fake_applescript(monkeypatch, [rows])
    result = calendar_mac.calendar_events("today")
    assert result.speech.startswith("You have 15 or more events today.")
    assert len(result.display["events"]) == 15


def test_calendar_events_rejects_bad_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_applescript(monkeypatch, [])
    with pytest.raises(ToolError, match="today, tomorrow, or week"):
        calendar_mac.calendar_events("month")


def test_calendar_events_skips_malformed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good = _row("Lunch", datetime.now() + timedelta(hours=1))
    _fake_applescript(monkeypatch, [f"garbage line\n{good}\nnot|||enough"])
    result = calendar_mac.calendar_events("today")
    assert isinstance(result, ToolResult)
    assert len(result.display["events"]) == 1


# -- create_event -------------------------------------------------------------

def test_create_event_builds_script(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_applescript(monkeypatch, [""])
    out = calendar_mac.create_event(
        "Lunch", "2026-07-20 12:30", duration_min=45, location="Cafe"
    )
    assert out == "Event created: Lunch, July 20 at 12:30 PM."
    script = calls[0]
    assert 'summary:"Lunch"' in script
    assert 'location:"Cafe"' in script
    assert "(45 * minutes)" in script
    assert "set year of theDate to 2026" in script
    assert "first calendar whose writable is true" in script


def test_create_event_default_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_applescript(monkeypatch, [""])
    calendar_mac.create_event("Call", "2026-07-20 09:00")
    assert "(60 * minutes)" in calls[0]
    assert "location:" not in calls[0]


def test_create_event_rejects_bad_start(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_applescript(monkeypatch, [])
    with pytest.raises(ToolError, match="couldn't understand the start time"):
        calendar_mac.create_event("Lunch", "next tuesday noon")


def test_create_event_rejects_bad_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_applescript(monkeypatch, [])
    with pytest.raises(ToolError, match="between 1 minute and 24 hours"):
        calendar_mac.create_event("Lunch", "2026-07-20 12:30", duration_min=0)
    with pytest.raises(ToolError, match="between 1 minute and 24 hours"):
        calendar_mac.create_event("Lunch", "2026-07-20 12:30", duration_min=9999)


def test_create_event_rejects_empty_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_applescript(monkeypatch, [])
    with pytest.raises(ToolError, match="call the event"):
        calendar_mac.create_event("  ", "2026-07-20 12:30")


# -- registration -------------------------------------------------------------

def test_calendar_tool_risks() -> None:
    events_spec = registry.get("calendar_events")
    assert events_spec is not None
    assert events_spec.risk == "safe"
    assert events_spec.category == "calendar"
    create_spec = registry.get("create_event")
    assert create_spec is not None
    assert create_spec.risk == "confirm"
    assert create_spec.category == "calendar"
    summary = create_spec.summary({"title": "Lunch", "start": "2026-07-20 12:30"})
    assert "Lunch" in summary
    assert "2026-07-20 12:30" in summary
