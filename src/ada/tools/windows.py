"""Window management tools built on System Events.

These require macOS Accessibility permission; run_applescript converts the
permission error into a helpful ToolError.
"""

from __future__ import annotations

import logging

from .applescript import PERMISSION_MESSAGE, applescript_quote, run_applescript
from .registry import ToolError, ToolResult, tool

log = logging.getLogger("ada.tools.windows")

_MENU_BAR_PX = 25
_SEP = "|||"

_LIST_SCRIPT = """\
tell application "System Events"
	set procNames to name of every application process whose visible is true
	set frontName to ""
	set frontWin to ""
	try
		set frontProc to first application process whose frontmost is true
		set frontName to name of frontProc
		try
			set frontWin to name of front window of frontProc
		end try
	end try
end tell
set AppleScript's text item delimiters to ", "
return frontName & "|||" & frontWin & "|||" & (procNames as text)"""

_ACTIONS = ("focus", "minimize", "maximize", "left_half", "right_half", "center", "fullscreen")


def _looks_like_missing_app(exc: ToolError) -> bool:
    """True when an osascript error means the process/app was not found.

    run_applescript strips the trailing "(-NNNN)" code, so we match on the
    surviving human text. Apostrophes are removed so "Can't"/"cant" and
    "isn't"/"isnt" both match.
    """
    text = str(exc).lower().replace("'", "").replace("’", "")
    return (
        "cant get application process" in text
        or "cant get application" in text
        or "isnt running" in text
    )


def _target_clause(app_name: str | None) -> str:
    if app_name:
        return f"application process {applescript_quote(app_name)}"
    return "(first application process whose frontmost is true)"


def _probe_target(app_name: str | None) -> tuple[str, int]:
    """Return (process name, window count) for the target process."""
    script = (
        'tell application "System Events"\n'
        f"\ttell {_target_clause(app_name)}\n"
        f'\t\treturn name & "{_SEP}" & (count of windows)\n'
        "\tend tell\n"
        "end tell"
    )
    try:
        out = run_applescript(script)
    except ToolError as exc:
        # Only rewrite to "isn't running" when the error actually says the
        # process wasn't found; permission/timeout/other errors pass through.
        if app_name and str(exc) != PERMISSION_MESSAGE and _looks_like_missing_app(exc):
            raise ToolError(f"{app_name} isn't running.") from None
        raise
    parts = out.split(_SEP)
    if len(parts) != 2:
        raise ToolError("I couldn't inspect that app's windows.")
    name = parts[0].strip() or (app_name or "That app")
    try:
        count = int(parts[1].strip())
    except ValueError:
        count = 0
    return name, count


def _desktop_area() -> tuple[int, int, int, int]:
    """Usable desktop area as (x, y, width, height), below the menu bar.

    Prefers AppKit's NSScreen (which reports the main screen's visible frame,
    excluding the menu bar and Dock) when pyobjc is available. Falls back to
    Finder's desktop bounds, which on multi-monitor setups is the union of all
    displays and so is only approximate.
    """
    try:
        from AppKit import NSScreen  # type: ignore[import-not-found]
    except ImportError:
        NSScreen = None  # type: ignore[assignment]

    if NSScreen is not None:
        screen = NSScreen.mainScreen()
        if screen is not None:
            visible = screen.visibleFrame()
            full = screen.frame()
            x = int(visible.origin.x)
            width = int(visible.size.width)
            height = int(visible.size.height)
            # Cocoa uses a bottom-left origin; System Events uses top-left.
            # visibleFrame already excludes the menu bar, so don't subtract it.
            top = int(full.size.height - (visible.origin.y + visible.size.height))
            return x, top, width, height

    out = run_applescript('tell application "Finder" to get bounds of window of desktop')
    parts = [p.strip() for p in out.split(",")]
    if len(parts) != 4:
        raise ToolError("I couldn't measure the screen.")
    try:
        x0, y0, x1, y1 = (int(float(p)) for p in parts)
    except ValueError:
        raise ToolError("I couldn't measure the screen.") from None
    top = y0 + _MENU_BAR_PX
    return x0, top, x1 - x0, y1 - top


def _set_frame(app_name: str | None, x: int, y: int, w: int, h: int) -> None:
    script = (
        'tell application "System Events"\n'
        f"\ttell {_target_clause(app_name)}\n"
        f"\t\tset position of front window to {{{x}, {y}}}\n"
        f"\t\tset size of front window to {{{w}, {h}}}\n"
        "\tend tell\n"
        "end tell"
    )
    run_applescript(script)


@tool(
    name="list_windows",
    description=(
        "List the currently visible applications and the frontmost app's "
        "front window. Use to find out what's open or what the user is "
        "looking at before acting on windows."
    ),
    input_schema={"type": "object", "properties": {}},
    risk="safe",
    category="windows",
)
def list_windows() -> str | ToolResult:
    out = run_applescript(_LIST_SCRIPT)
    parts = out.split(_SEP, 2)
    while len(parts) < 3:
        parts.append("")
    front, front_win, procs = (p.strip() for p in parts)
    if not procs:
        return "I don't see any open apps."
    summary = f"Open apps: {procs}."
    if front and front_win:
        summary += f" Frontmost: {front}, showing {front_win}."
    elif front:
        summary += f" Frontmost: {front}, with no open windows."
    items: list[dict] = []
    for name in procs.split(", "):
        name = name.strip()
        if not name:
            continue
        item: dict = {"text": name}
        if name == front and front_win:
            item["sub"] = front_win
        items.append(item)
    return ToolResult(
        speech=summary,
        display={"kind": "list", "title": "Open apps", "items": items},
    )


@tool(
    name="window_action",
    description=(
        "Control application windows: focus an app, minimize, maximize, snap "
        "to the left or right half of the screen, center, or toggle "
        "fullscreen. Acts on the frontmost app unless app_name is given."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
                "description": "What to do with the window.",
            },
            "app_name": {
                "type": "string",
                "description": "The app whose window to act on; omit for the frontmost app.",
            },
        },
        "required": ["action"],
    },
    risk="safe",
    category="windows",
)
def window_action(action: str, app_name: str | None = None) -> str:
    if action not in _ACTIONS:
        raise ToolError(f"I don't know the window action {action}.")

    if action == "focus":
        if not app_name:
            raise ToolError("Tell me which app to focus.")
        try:
            run_applescript(f"tell application {applescript_quote(app_name)} to activate")
        except ToolError as exc:
            # Only rewrite to "couldn't find" when the app truly wasn't found;
            # permission/timeout/other errors pass through unchanged.
            if str(exc) != PERMISSION_MESSAGE and _looks_like_missing_app(exc):
                raise ToolError(f"I couldn't find an app called {app_name}.") from None
            raise
        return f"Focused {app_name}."

    name, window_count = _probe_target(app_name)
    if window_count == 0:
        raise ToolError(f"{name} has no open windows.")

    if action == "minimize":
        script = (
            'tell application "System Events"\n'
            f"\ttell {_target_clause(app_name)}\n"
            '\t\tset value of attribute "AXMinimized" of front window to true\n'
            "\tend tell\n"
            "end tell"
        )
        run_applescript(script)
        return f"Minimized {name}."

    if action == "fullscreen":
        script = (
            'tell application "System Events"\n'
            f"\ttell {_target_clause(app_name)}\n"
            "\t\ttell front window\n"
            '\t\t\tset wasFull to value of attribute "AXFullScreen"\n'
            '\t\t\tset value of attribute "AXFullScreen" to not wasFull\n'
            "\t\t\treturn (not wasFull) as text\n"
            "\t\tend tell\n"
            "\tend tell\n"
            "end tell"
        )
        out = run_applescript(script)
        if out.strip().lower() == "true":
            return f"Made {name} fullscreen."
        return f"Took {name} out of fullscreen."

    x, y, width, height = _desktop_area()
    if action == "maximize":
        _set_frame(app_name, x, y, width, height)
        return f"Maximized {name}."
    if action == "left_half":
        _set_frame(app_name, x, y, width // 2, height)
        return f"Moved {name} to the left half."
    if action == "right_half":
        _set_frame(app_name, x + width // 2, y, width - width // 2, height)
        return f"Moved {name} to the right half."
    # center: 2/3-size window centered in the usable area
    w = max(400, width * 2 // 3)
    h = max(300, height * 2 // 3)
    w, h = min(w, width), min(h, height)
    _set_frame(app_name, x + (width - w) // 2, y + (height - h) // 2, w, h)
    return f"Centered {name}."
