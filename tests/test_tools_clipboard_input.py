"""Tests for ada.tools.clipboard (pbpaste/pbcopy) and ada.tools.input_text.

subprocess and AppleScript are mocked at their import sites.
"""

from __future__ import annotations

import importlib
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from ada.tools import registry
from ada.tools.registry import ToolError, ToolResult

clipboard = importlib.import_module("ada.tools.clipboard")
input_text_mod = importlib.import_module("ada.tools.input_text")


def _install_fake_pasteboard(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str = "",
    returncode: int = 0,
) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    monkeypatch.setattr("ada.tools.clipboard.subprocess.run", fake_run)
    return seen


# -- read_clipboard -----------------------------------------------------------

def test_read_clipboard_short_text(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_fake_pasteboard(monkeypatch, stdout="hello world")
    result = clipboard.read_clipboard()
    assert isinstance(result, ToolResult)
    assert result.speech == "hello world"
    assert result.display == {
        "kind": "markdown",
        "title": "Clipboard",
        "body": "hello world",
    }
    assert seen["cmd"] == ["/usr/bin/pbpaste"]
    assert seen["kwargs"]["timeout"] == 5


def test_read_clipboard_long_text_truncates_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "word " * 60  # 300 chars
    _install_fake_pasteboard(monkeypatch, stdout=text)
    result = clipboard.read_clipboard()
    assert result.speech.endswith("…")
    assert len(result.speech) <= 81
    assert result.display["body"] == text.strip()  # card keeps the full text


def test_read_clipboard_caps_card_body(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pasteboard(monkeypatch, stdout="x" * 5000)
    result = clipboard.read_clipboard()
    assert len(result.display["body"]) <= 4001
    assert result.display["body"].endswith("…")


def test_read_clipboard_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pasteboard(monkeypatch, stdout="  \n ")
    assert clipboard.read_clipboard() == "The clipboard is empty."


def test_read_clipboard_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pasteboard(monkeypatch, returncode=1)
    with pytest.raises(ToolError, match="couldn't read the clipboard"):
        clipboard.read_clipboard()


def test_read_clipboard_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_timeout(*args: Any, **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="pbpaste", timeout=5)

    monkeypatch.setattr("ada.tools.clipboard.subprocess.run", raise_timeout)
    with pytest.raises(ToolError, match="timed out"):
        clipboard.read_clipboard()


# -- set_clipboard ------------------------------------------------------------

def test_set_clipboard(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_fake_pasteboard(monkeypatch)
    assert clipboard.set_clipboard("copy this") == "Copied."
    assert seen["cmd"] == ["/usr/bin/pbcopy"]
    assert seen["kwargs"]["input"] == "copy this"


def test_set_clipboard_rejects_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pasteboard(monkeypatch)
    with pytest.raises(ToolError, match="what to copy"):
        clipboard.set_clipboard("")


def test_set_clipboard_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pasteboard(monkeypatch, returncode=1)
    with pytest.raises(ToolError, match="couldn't write"):
        clipboard.set_clipboard("text")


# -- type_text ----------------------------------------------------------------

def _fake_applescript(
    monkeypatch: pytest.MonkeyPatch, responses: list[object] | None = None
) -> list[str]:
    calls: list[str] = []
    queue = list(responses or [])

    def fake(script: str, timeout: float = 10.0) -> str:
        calls.append(script)
        result = queue.pop(0) if queue else ""
        if isinstance(result, Exception):
            raise result
        return str(result)

    monkeypatch.setattr("ada.tools.input_text.run_applescript", fake)
    return calls


def test_type_text_escapes_and_handles_newlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_applescript(monkeypatch)
    assert input_text_mod.type_text('He said "hi" \\o/\nBye') == "Typed."
    script = calls[0]
    assert script.startswith('tell application "System Events"')
    # Quotes and backslashes are escaped for AppleScript.
    assert 'keystroke "He said \\"hi\\" \\\\o/"' in script
    assert "keystroke return" in script
    assert 'keystroke "Bye"' in script


def test_type_text_blank_lines_press_return_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_applescript(monkeypatch)
    input_text_mod.type_text("a\n\nb")
    assert calls[0].count("keystroke return") == 2
    assert calls[0].count('keystroke "') == 2  # no keystroke "" emitted


def test_type_text_rejects_long_text(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_applescript(monkeypatch)
    with pytest.raises(ToolError, match="too much text"):
        input_text_mod.type_text("x" * 1001)
    assert calls == []


def test_type_text_rejects_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_applescript(monkeypatch)
    with pytest.raises(ToolError, match="what to type"):
        input_text_mod.type_text("")


# -- registration -------------------------------------------------------------

def test_clipboard_tools_registered_safe() -> None:
    for name in ("read_clipboard", "set_clipboard"):
        spec = registry.get(name)
        assert spec is not None
        assert spec.risk == "safe"
        assert spec.category == "clipboard"
    # The model is warned that clipboard contents can be sensitive.
    assert "sensitive" in registry.get("read_clipboard").description


def test_type_text_confirmation_shows_exact_text() -> None:
    spec = registry.get("type_text")
    assert spec is not None
    assert spec.risk == "confirm"  # the dialog IS the safety design
    assert spec.category == "input"
    # The exact text appears in the confirmation summary...
    assert spec.summary({"text": "rm -rf notes"}) == (
        'Type into the front app: "rm -rf notes"'
    )
    # ...truncated at 140 characters for display.
    summary = spec.summary({"text": "y" * 300})
    assert "y" * 140 + "…" in summary
    assert "y" * 141 not in summary
