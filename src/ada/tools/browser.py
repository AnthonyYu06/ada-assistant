"""Browser tools: tabs and pages in Safari or Google Chrome.

Both browsers speak the same AppleScript verbs (modulo `name`/`title` and
how the active tab is addressed). Each tool targets the running browser,
preferring the frontmost one when both are open.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

from .applescript import PERMISSION_MESSAGE, run_applescript
from .registry import ToolError, ToolResult, tool

log = logging.getLogger("ada.tools.browser")

_SEP = "|||"
_DISPLAY_CAP = 20  # tabs shown on the card
_ENUM_CAP = 100  # tabs enumerated for matching (keeps scripts fast)

_SAFARI = "Safari"
_CHROME = "Google Chrome"

_BROWSER_CHECK = """\
tell application "System Events"
	set hasSafari to exists process "Safari"
	set hasChrome to exists process "Google Chrome"
	set frontName to ""
	try
		set frontName to name of first application process whose frontmost is true
	end try
end tell
return (hasSafari as text) & "|||" & (hasChrome as text) & "|||" & frontName"""


@dataclass
class _Tab:
    window: int
    index: int
    title: str
    url: str


def _short_name(browser: str) -> str:
    return "Chrome" if browser == _CHROME else browser


def _domain(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.")


def _pick_browser() -> str:
    """Return the browser app to target, preferring the frontmost one."""
    out = run_applescript(_BROWSER_CHECK)
    parts = out.split(_SEP)
    while len(parts) < 3:
        parts.append("")
    has_safari = parts[0].strip().lower() == "true"
    has_chrome = parts[1].strip().lower() == "true"
    front = parts[2].strip()
    if front == _SAFARI and has_safari:
        return _SAFARI
    if front == _CHROME and has_chrome:
        return _CHROME
    if has_safari:
        return _SAFARI
    if has_chrome:
        return _CHROME
    raise ToolError("No browser is open.")


def _tabs_script(browser: str, cap: int) -> str:
    title_prop = "name" if browser == _SAFARI else "title"
    return "\n".join(
        [
            f'set sep to "{_SEP}"',
            "set outLines to {}",
            "set totalTabs to 0",
            "set collected to 0",
            f'tell application "{browser}"',
            "\tset winCount to count of windows",
            "\trepeat with wIdx from 1 to winCount",
            "\t\tset totalTabs to totalTabs + (count of tabs of window wIdx)",
            "\tend repeat",
            "\trepeat with wIdx from 1 to winCount",
            "\t\tset tabCount to count of tabs of window wIdx",
            "\t\trepeat with tIdx from 1 to tabCount",
            f"\t\t\tif collected is greater than or equal to {cap} then exit repeat",
            "\t\t\tset t to tab tIdx of window wIdx",
            '\t\t\tset tTitle to ""',
            "\t\t\ttry",
            f"\t\t\t\tset tTitle to {title_prop} of t",
            "\t\t\tend try",
            '\t\t\tif tTitle is missing value then set tTitle to ""',
            '\t\t\tset tUrl to ""',
            "\t\t\ttry",
            "\t\t\t\tset tUrl to URL of t",
            "\t\t\tend try",
            '\t\t\tif tUrl is missing value then set tUrl to ""',
            "\t\t\tset end of outLines to (wIdx as text) & sep & (tIdx as text) "
            "& sep & tTitle & sep & tUrl",
            "\t\t\tset collected to collected + 1",
            "\t\tend repeat",
            f"\t\tif collected is greater than or equal to {cap} then exit repeat",
            "\tend repeat",
            "end tell",
            "set AppleScript's text item delimiters to linefeed",
            "return (totalTabs as text) & linefeed & (outLines as text)",
        ]
    )


def _list_tabs(browser: str, cap: int) -> tuple[int, list[_Tab]]:
    """Return (total tab count, enumerated tabs capped at `cap`)."""
    out = run_applescript(_tabs_script(browser, cap), timeout=15.0)
    lines = out.splitlines()
    if not lines:
        return 0, []
    try:
        total = int(lines[0].strip())
    except ValueError:
        total = 0
    tabs: list[_Tab] = []
    for line in lines[1:]:
        parts = line.split(_SEP)
        if len(parts) != 4:
            continue
        try:
            window, index = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        tabs.append(_Tab(window, index, parts[2].strip(), parts[3].strip()))
    return max(total, len(tabs)), tabs


def _find_tab(match: str, tabs: list[_Tab]) -> _Tab:
    """Match a tab by 1-based position in the listing or a title substring."""
    match = (match or "").strip()
    if not match:
        raise ToolError("Tell me which tab — a number or part of its title.")
    if match.lstrip("-").isdigit():
        position = int(match)
        if not 1 <= position <= len(tabs):
            raise ToolError(f"There's no tab number {position}.")
        return tabs[position - 1]
    lowered = match.lower()
    for tab in tabs:
        if lowered in tab.title.lower():
            return tab
    raise ToolError(f"I couldn't find a tab matching {match}.")


def _tab_label(tab: _Tab) -> str:
    label = tab.title or _domain(tab.url) or "that tab"
    if len(label) > 60:
        label = label[:57].rstrip() + "…"
    return label


@tool(
    name="browser_tabs",
    description=(
        "List the tabs open in the user's browser (Safari or Chrome), with "
        "titles and sites. Use when the user asks what tabs are open or "
        "before switching to or closing a tab."
    ),
    input_schema={"type": "object", "properties": {}},
    risk="safe",
    category="browser",
)
def browser_tabs() -> str | ToolResult:
    browser = _pick_browser()
    total, tabs = _list_tabs(browser, _DISPLAY_CAP)
    if total == 0:
        return f"{_short_name(browser)} has no tabs open."
    items: list[dict] = []
    for tab in tabs[:_DISPLAY_CAP]:
        item: dict = {"text": tab.title or _domain(tab.url) or "Untitled"}
        domain = _domain(tab.url)
        if domain:
            item["sub"] = domain
        items.append(item)
    if total > len(items):
        items.append({"text": f"+{total - len(items)} more"})
    noun = "tab" if total == 1 else "tabs"
    return ToolResult(
        speech=f"You have {total} {noun} open in {_short_name(browser)}.",
        display={
            "kind": "list",
            "title": f"{_short_name(browser)} tabs",
            "items": items,
        },
    )


@tool(
    name="current_page",
    description=(
        "Get the title and URL of the page in the browser's front tab "
        "(Safari or Chrome). Use when the user asks about 'this page' or "
        "what they're reading."
    ),
    input_schema={"type": "object", "properties": {}},
    risk="safe",
    category="browser",
)
def current_page() -> ToolResult:
    browser = _pick_browser()
    if browser == _SAFARI:
        script = (
            'tell application "Safari" to return (name of front document) '
            f'& "{_SEP}" & (URL of front document)'
        )
    else:
        script = (
            'tell application "Google Chrome" to return '
            "(title of active tab of front window) "
            f'& "{_SEP}" & (URL of active tab of front window)'
        )
    try:
        out = run_applescript(script)
    except ToolError as exc:
        if str(exc) == PERMISSION_MESSAGE:
            raise
        raise ToolError(f"{_short_name(browser)} has no open page.") from None
    parts = out.split(_SEP)
    title = parts[0].strip() if parts else ""
    url = parts[1].strip() if len(parts) > 1 else ""
    label = title or _domain(url) or "an untitled page"
    if len(label) > 60:
        label = label[:57].rstrip() + "…"
    pairs = [["Title", title or "Untitled"], ["URL", url]]
    return ToolResult(
        speech=f"You're on {label}.",
        display={"kind": "keyvalue", "title": "Current page", "pairs": pairs},
    )


@tool(
    name="switch_tab",
    description=(
        "Switch the browser (Safari or Chrome) to a different open tab, "
        "matched by its number in the tab list or by part of its title. Use "
        "when the user asks to go to or focus a particular tab."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "match": {
                "type": "string",
                "description": "A 1-based tab number or a fragment of the tab's title.",
            }
        },
        "required": ["match"],
    },
    risk="safe",
    category="browser",
)
def switch_tab(match: str) -> str:
    browser = _pick_browser()
    _total, tabs = _list_tabs(browser, _ENUM_CAP)
    if not tabs:
        raise ToolError(f"{_short_name(browser)} has no tabs open.")
    tab = _find_tab(match, tabs)
    if browser == _SAFARI:
        script = (
            'tell application "Safari"\n'
            "\tactivate\n"
            f"\tset current tab of window {tab.window} to tab {tab.index} "
            f"of window {tab.window}\n"
            f"\tset index of window {tab.window} to 1\n"
            "end tell"
        )
    else:
        script = (
            'tell application "Google Chrome"\n'
            "\tactivate\n"
            f"\tset active tab index of window {tab.window} to {tab.index}\n"
            f"\tset index of window {tab.window} to 1\n"
            "end tell"
        )
    run_applescript(script)
    return f"Switched to {_tab_label(tab)}."


@tool(
    name="close_tab",
    description=(
        "Close one open browser tab (Safari or Chrome), matched by its "
        "number in the tab list or by part of its title. Use only when the "
        "user asks to close a tab."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "match": {
                "type": "string",
                "description": "A 1-based tab number or a fragment of the tab's title.",
            }
        },
        "required": ["match"],
    },
    risk="confirm",
    category="browser",
    describe_call=lambda a: f"Close the browser tab matching '{a['match']}'",
)
def close_tab(match: str) -> str:
    browser = _pick_browser()
    _total, tabs = _list_tabs(browser, _ENUM_CAP)
    if not tabs:
        raise ToolError(f"{_short_name(browser)} has no tabs open.")
    tab = _find_tab(match, tabs)
    run_applescript(
        f'tell application "{browser}" to close tab {tab.index} '
        f"of window {tab.window}"
    )
    return f"Closed {_tab_label(tab)}."
