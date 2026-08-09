"""Tests for ada.tools.context: front-app info and screenshots.

AppleScript is mocked at the import site; screencapture/sips via subprocess.run.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ada.tools import registry
from ada.tools.registry import ToolError, ToolResult

context = importlib.import_module("ada.tools.context")


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

    monkeypatch.setattr("ada.tools.context.run_applescript", fake)
    return calls


# -- front_app_info -----------------------------------------------------------

def test_front_app_info_plain_app(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_applescript(monkeypatch, ["Notes|||Groceries"])
    result = context.front_app_info()
    assert isinstance(result, ToolResult)
    assert result.speech == "You're in Notes, on Groceries."
    assert result.display["kind"] == "keyvalue"
    assert result.display["pairs"] == [["App", "Notes"], ["Window", "Groceries"]]
    assert len(calls) == 1  # non-browser: no page lookup


def test_front_app_info_includes_browser_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_applescript(
        monkeypatch,
        ["Safari|||Apple – Start", "Apple|||https://www.apple.com/"],
    )
    result = context.front_app_info()
    pairs = result.display["pairs"]
    assert ["App", "Safari"] in pairs
    assert ["Page", "Apple"] in pairs
    assert ["URL", "https://www.apple.com/"] in pairs
    assert "apple.com" not in result.speech  # URLs are never spoken


def test_front_app_info_page_error_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_applescript(
        monkeypatch, ["Google Chrome|||New Tab", ToolError("no window")]
    )
    result = context.front_app_info()
    assert result.display["pairs"] == [
        ["App", "Google Chrome"],
        ["Window", "New Tab"],
    ]


def test_front_app_info_no_front_app(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_applescript(monkeypatch, ["|||"])
    with pytest.raises(ToolError, match="couldn't see"):
        context.front_app_info()


# -- capture_screen -----------------------------------------------------------

def _install_fake_capture(
    monkeypatch: pytest.MonkeyPatch,
    jpeg: bytes = b"\xff\xd8fake-jpeg-bytes",
    capture_rc: int = 0,
) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        if cmd[0].endswith("screencapture"):
            seen["path"] = cmd[-1]
            seen["capture_cmd"] = cmd
            if capture_rc == 0:
                Path(cmd[-1]).write_bytes(jpeg)
            return SimpleNamespace(returncode=capture_rc, stdout="", stderr="boom")
        if cmd[0].endswith("sips"):
            seen["sips_cmd"] = cmd
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr("ada.tools.context.subprocess.run", fake_run)
    return seen


def test_capture_screen_returns_image_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _install_fake_capture(monkeypatch)
    result = context.capture_screen()
    assert isinstance(result, ToolResult)
    assert result.speech == "Here's your screen."
    assert result.display["kind"] == "image"
    assert result.display["caption"] == "Screenshot"
    assert result.display["src"].startswith("data:image/jpeg;base64,")
    # Non-interactive jpg capture, downscaled with sips.
    assert "-x" in seen["capture_cmd"]
    assert seen["sips_cmd"][:2] == ["/usr/bin/sips", "--resampleWidth"]
    # The temp file is always deleted.
    assert not os.path.exists(seen["path"])


def test_capture_screen_failure_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _install_fake_capture(monkeypatch, capture_rc=1)
    with pytest.raises(ToolError, match="Screen Recording"):
        context.capture_screen()
    assert not os.path.exists(seen["path"])


def test_capture_screen_rejects_oversized_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _install_fake_capture(monkeypatch, jpeg=b"\xff" * 3_200_000)
    with pytest.raises(ToolError, match="too large"):
        context.capture_screen()
    assert not os.path.exists(seen["path"])


# -- registration -------------------------------------------------------------

def test_context_tool_risks() -> None:
    info_spec = registry.get("front_app_info")
    assert info_spec is not None
    assert info_spec.risk == "safe"
    assert info_spec.category == "context"
    capture_spec = registry.get("capture_screen")
    assert capture_spec is not None
    assert capture_spec.risk == "confirm"  # screen contents are sensitive
    assert capture_spec.category == "context"
    assert "screenshot" in capture_spec.summary({}).lower()
