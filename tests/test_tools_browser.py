"""Tests for ada.tools.browser: browser picking, tabs, switching, closing.

All AppleScript is mocked at the import site (ada.tools.browser.run_applescript).
"""

from __future__ import annotations

import importlib

import pytest

from ada.tools import registry
from ada.tools.registry import ToolError, ToolResult

browser = importlib.import_module("ada.tools.browser")

_SAFARI_FRONT = "true|||false|||Safari"
_CHROME_ONLY = "false|||true|||Finder"
_NONE_OPEN = "false|||false|||Finder"

# total-count line, then one line per tab: window|||tab|||title|||url
_TABS_LISTING = (
    "3\n"
    "1|||1|||Title A|||https://www.a.com/x\n"
    "1|||2|||Title B|||https://b.org/page\n"
    "2|||1|||Title C|||"
)


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

    monkeypatch.setattr("ada.tools.browser.run_applescript", fake)
    return calls


# -- browser picking ----------------------------------------------------------

def test_no_browser_open(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_applescript(monkeypatch, [_NONE_OPEN])
    with pytest.raises(ToolError, match="No browser is open"):
        browser.browser_tabs()


def test_prefers_frontmost_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both open, Chrome frontmost -> Chrome scripts.
    calls = _fake_applescript(
        monkeypatch,
        ["true|||true|||Google Chrome", "1\n1|||1|||T|||https://x.com"],
    )
    result = browser.browser_tabs()
    assert 'tell application "Google Chrome"' in calls[1]
    assert "title of t" in calls[1]
    assert "Chrome" in result.speech


# -- browser_tabs -------------------------------------------------------------

def test_browser_tabs_list_card(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_applescript(monkeypatch, [_SAFARI_FRONT, _TABS_LISTING])
    result = browser.browser_tabs()
    assert isinstance(result, ToolResult)
    assert result.speech == "You have 3 tabs open in Safari."
    assert 'tell application "Safari"' in calls[1]
    assert "name of t" in calls[1]
    card = result.display
    assert card["kind"] == "list"
    assert card["title"] == "Safari tabs"
    assert card["items"][0] == {"text": "Title A", "sub": "a.com"}
    assert card["items"][1] == {"text": "Title B", "sub": "b.org"}
    assert card["items"][2] == {"text": "Title C"}  # no URL -> no sub


def test_browser_tabs_notes_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = "\n".join(
        f"1|||{i}|||Tab {i}|||https://site{i}.com" for i in range(1, 21)
    )
    _fake_applescript(monkeypatch, [_SAFARI_FRONT, f"25\n{rows}"])
    result = browser.browser_tabs()
    assert result.speech == "You have 25 tabs open in Safari."
    items = result.display["items"]
    assert len(items) == 21
    assert items[-1] == {"text": "+5 more"}


def test_browser_tabs_none_open(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_applescript(monkeypatch, [_SAFARI_FRONT, "0\n"])
    assert browser.browser_tabs() == "Safari has no tabs open."


# -- current_page -------------------------------------------------------------

def test_current_page_keyvalue_card(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_applescript(
        monkeypatch, [_SAFARI_FRONT, "Apple|||https://www.apple.com/"]
    )
    result = browser.current_page()
    assert isinstance(result, ToolResult)
    assert result.speech == "You're on Apple."
    assert result.display["kind"] == "keyvalue"
    assert ["Title", "Apple"] in result.display["pairs"]
    assert ["URL", "https://www.apple.com/"] in result.display["pairs"]
    # The URL is never read aloud.
    assert "apple.com" not in result.speech


def test_current_page_no_page(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_applescript(
        monkeypatch, [_SAFARI_FRONT, ToolError("missing value etc.")]
    )
    with pytest.raises(ToolError, match="no open page"):
        browser.current_page()


# -- switch_tab ---------------------------------------------------------------

def test_switch_tab_by_index_safari(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_applescript(monkeypatch, [_SAFARI_FRONT, _TABS_LISTING, ""])
    assert browser.switch_tab("2") == "Switched to Title B."
    assert "set current tab of window 1 to tab 2 of window 1" in calls[2]


def test_switch_tab_by_title_chrome(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_applescript(monkeypatch, [_CHROME_ONLY, _TABS_LISTING, ""])
    assert browser.switch_tab("title c") == "Switched to Title C."
    assert "set active tab index of window 2 to 1" in calls[2]


def test_switch_tab_bad_index(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_applescript(monkeypatch, [_SAFARI_FRONT, _TABS_LISTING])
    with pytest.raises(ToolError, match="no tab number 9"):
        browser.switch_tab("9")


def test_switch_tab_no_match(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_applescript(monkeypatch, [_SAFARI_FRONT, _TABS_LISTING])
    with pytest.raises(ToolError, match="couldn't find a tab"):
        browser.switch_tab("zebra")


# -- close_tab ----------------------------------------------------------------

def test_close_tab(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_applescript(monkeypatch, [_SAFARI_FRONT, _TABS_LISTING, ""])
    assert browser.close_tab("Title A") == "Closed Title A."
    assert "close tab 1 of window 1" in calls[2]


def test_close_tab_no_tabs(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_applescript(monkeypatch, [_SAFARI_FRONT, "0\n"])
    with pytest.raises(ToolError, match="no tabs open"):
        browser.close_tab("anything")


# -- registration -------------------------------------------------------------

def test_browser_tool_risks() -> None:
    for name in ("browser_tabs", "current_page", "switch_tab"):
        spec = registry.get(name)
        assert spec is not None
        assert spec.risk == "safe"
        assert spec.category == "browser"
    close_spec = registry.get("close_tab")
    assert close_spec is not None
    assert close_spec.risk == "confirm"
    assert close_spec.category == "browser"
    assert "Docs" in close_spec.summary({"match": "Docs"})
