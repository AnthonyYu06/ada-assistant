"""Tests for ada.tools.applescript: quoting, dates, osascript error mapping."""

from __future__ import annotations

import subprocess
import types
from typing import Any

import pytest

from ada.tools import applescript as applescript_mod
from ada.tools.applescript import (
    PERMISSION_MESSAGE,
    applescript_date,
    applescript_quote,
    run_applescript,
)
from ada.tools.registry import ToolError


# -- applescript_quote --------------------------------------------------------

def test_quote_plain_string() -> None:
    assert applescript_quote("hello world") == '"hello world"'


def test_quote_escapes_double_quotes() -> None:
    assert applescript_quote('say "hi"') == r'"say \"hi\""'


def test_quote_escapes_backslashes() -> None:
    assert applescript_quote(r"back\slash") == r'"back\\slash"'


def test_quote_escapes_backslash_before_quote() -> None:
    # Backslashes must be doubled BEFORE quotes are escaped, or the escape
    # of the quote itself would get mangled.
    assert applescript_quote(r'both\ and "q"') == r'"both\\ and \"q\""'


def test_quote_empty_string() -> None:
    assert applescript_quote("") == '""'


# -- applescript_date ---------------------------------------------------------

def test_date_contains_components() -> None:
    snippet = applescript_date("2026-07-13T15:42")
    assert "set year of theDate to 2026" in snippet
    assert "set month of theDate to 7" in snippet
    assert "set day of theDate to 13" in snippet
    assert "set time of theDate to (15 * hours + 42 * minutes + 0)" in snippet
    # Day is reset to 1 before the month changes so switching months can
    # never overflow (e.g. Jan 31 -> February).
    lines = snippet.splitlines()
    assert lines[0] == "set theDate to current date"
    assert lines[1] == "set day of theDate to 1"
    assert lines.index("set day of theDate to 1") < lines.index(
        "set month of theDate to 7"
    )


def test_date_with_seconds() -> None:
    snippet = applescript_date("2026-07-13T15:42:30")
    assert "set time of theDate to (15 * hours + 42 * minutes + 30)" in snippet


@pytest.mark.parametrize(
    "garbage",
    ["next tuesday", "", "2026-13-45T99:99", "13/07/2026", None, "tomorrow at 5"],
)
def test_date_rejects_garbage(garbage: Any) -> None:
    with pytest.raises(ToolError, match="couldn't understand"):
        applescript_date(garbage)


# -- run_applescript ----------------------------------------------------------

def _patch_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
    raises: BaseException | None = None,
) -> list[list[str]]:
    """Replace the module's subprocess binding; returns the captured argv list."""
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        calls.append(list(cmd))
        if raises is not None:
            raise raises
        return types.SimpleNamespace(
            returncode=returncode, stdout=stdout, stderr=stderr
        )

    monkeypatch.setattr(
        applescript_mod,
        "subprocess",
        types.SimpleNamespace(
            run=fake_run, TimeoutExpired=subprocess.TimeoutExpired
        ),
    )
    return calls


def test_run_returns_stripped_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_run(monkeypatch, returncode=0, stdout="  Safari\n")
    result = run_applescript('tell application "System Events" to get name')
    assert result == "Safari"
    assert calls == [
        ["/usr/bin/osascript", "-e", 'tell application "System Events" to get name']
    ]


def test_run_maps_assistive_access_to_friendly_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_run(
        monkeypatch,
        returncode=1,
        stderr=(
            "61:75: execution error: System Events got an error: osascript "
            "is not allowed assistive access. (-25211)"
        ),
    )
    with pytest.raises(ToolError) as excinfo:
        run_applescript("tell application \"System Events\" to keystroke tab")
    assert str(excinfo.value) == PERMISSION_MESSAGE


def test_run_maps_automation_denied_code_to_friendly_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_run(
        monkeypatch,
        returncode=1,
        stderr="30:40: execution error: Not authorized to send Apple events. (-1719)",
    )
    with pytest.raises(ToolError) as excinfo:
        run_applescript("tell application \"Finder\" to activate")
    assert str(excinfo.value) == PERMISSION_MESSAGE


def test_run_maps_automation_consent_1743_to_friendly_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_run(
        monkeypatch,
        returncode=1,
        stderr=(
            "0:0: execution error: Not authorized to send Apple events "
            "to Finder. (-1743)"
        ),
    )
    with pytest.raises(ToolError) as excinfo:
        run_applescript('tell application "Finder" to activate')
    assert str(excinfo.value) == PERMISSION_MESSAGE


def test_run_cleans_generic_osascript_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run(
        monkeypatch,
        returncode=1,
        stderr=(
            "45:52: execution error: Safari got an error: "
            "it doesn't understand. (-1728)"
        ),
    )
    with pytest.raises(ToolError) as excinfo:
        run_applescript("bogus")
    assert str(excinfo.value) == (
        "That didn't work: Safari got an error: it doesn't understand."
    )


def test_run_nonzero_with_empty_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run(monkeypatch, returncode=2, stderr="")
    with pytest.raises(ToolError, match="unknown AppleScript error"):
        run_applescript("bogus")


def test_run_timeout_maps_to_tool_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run(
        monkeypatch,
        raises=subprocess.TimeoutExpired(cmd="osascript", timeout=10.0),
    )
    with pytest.raises(ToolError, match="timed out"):
        run_applescript("delay 9999")
