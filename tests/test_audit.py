"""Tests for ada.safety.audit.AuditLog: JSONL shape, truncation, no-op mode."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ada.safety.audit import AuditLog


def _read_events(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_jsonl_lines_parse(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    audit.record("startup", version="0.1.0")
    audit.record(
        "tool_call",
        tool="open_app",
        args={"app_name": "Safari"},
        allowed=True,
    )
    events = _read_events(tmp_path / "audit.jsonl")
    assert len(events) == 2
    assert events[0]["kind"] == "startup"
    assert events[0]["version"] == "0.1.0"
    assert events[1]["kind"] == "tool_call"
    assert events[1]["args"] == {"app_name": "Safari"}
    assert events[1]["allowed"] is True
    for event in events:
        # ts is valid UTC ISO 8601.
        stamp = datetime.fromisoformat(event["ts"])
        assert stamp.tzinfo is not None


def test_long_values_truncated(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    long = "x" * 500
    audit.record("transcript", text=long)
    (event,) = _read_events(tmp_path / "audit.jsonl")
    assert len(event["text"]) == 301  # 300 chars + ellipsis
    assert event["text"].endswith("…")
    assert event["text"][:300] == "x" * 300


def test_truncation_recurses_into_containers(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    long = "y" * 400
    audit.record("event", args={"note": long, "items": [long, "short"]}, n=7)
    (event,) = _read_events(tmp_path / "audit.jsonl")
    assert event["args"]["note"].endswith("…")
    assert event["args"]["items"][0].endswith("…")
    assert event["args"]["items"][1] == "short"
    assert event["n"] == 7  # non-strings untouched


def test_short_values_untouched(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    audit.record("event", text="short and sweet")
    (event,) = _read_events(tmp_path / "audit.jsonl")
    assert event["text"] == "short and sweet"


def test_disabled_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "audit.jsonl"
    audit = AuditLog(path, enabled=False)
    audit.record("startup")
    audit.record("tool_call", tool="open_app")
    assert not path.exists()
    # Disabled mode doesn't even create the directory.
    assert not path.parent.exists()


def test_non_serializable_values_coerced(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")

    class Widget:
        def __str__(self) -> str:
            return "<widget-42>"

    audit.record(
        "event",
        obj=Widget(),
        path=Path("/tmp/somewhere"),
        when=datetime(2026, 7, 13, 15, 42),
    )
    (event,) = _read_events(tmp_path / "audit.jsonl")
    assert event["obj"] == "<widget-42>"
    assert event["path"] == "/tmp/somewhere"
    assert "2026-07-13" in event["when"]


def test_write_failure_is_swallowed(tmp_path: Path) -> None:
    # Point the log at a directory: appending will fail, but record() must
    # never raise (the audit log can't take the assistant down).
    audit = AuditLog(tmp_path)
    audit.record("event", detail="this write fails silently")


def test_appends_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    AuditLog(path).record("first")
    AuditLog(path).record("second")
    events = _read_events(path)
    assert [e["kind"] for e in events] == ["first", "second"]


def test_parent_directory_created(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "audit.jsonl"
    audit = AuditLog(path)
    audit.record("event")
    assert path.is_file()
    assert len(_read_events(path)) == 1
