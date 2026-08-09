"""Context tools: what the user is looking at, and on-demand screenshots."""

from __future__ import annotations

import base64
import logging
import os
import subprocess
import tempfile

from .applescript import PERMISSION_MESSAGE, run_applescript
from .registry import ToolError, ToolResult, tool

log = logging.getLogger("ada.tools.context")

_SEP = "|||"
_MAX_DATA_URI_CHARS = 4_000_000

_FRONT_APP_SCRIPT = """\
tell application "System Events"
	set frontName to ""
	set winTitle to ""
	try
		set frontProc to first application process whose frontmost is true
		set frontName to name of frontProc
		try
			set winTitle to name of front window of frontProc
		end try
	end try
end tell
return frontName & "|||" & winTitle"""

_BROWSER_PAGE_SCRIPTS = {
    "Safari": (
        'tell application "Safari" to return (name of front document) '
        '& "|||" & (URL of front document)'
    ),
    "Google Chrome": (
        'tell application "Google Chrome" to return '
        '(title of active tab of front window) '
        '& "|||" & (URL of active tab of front window)'
    ),
}


@tool(
    name="front_app_info",
    description=(
        "What is the user looking at right now: the frontmost app and its "
        "front window title, plus the current page title and URL when the "
        "front app is Safari or Chrome. Use whenever the user says 'this', "
        "'this app', 'this page', or asks what they're looking at."
    ),
    input_schema={"type": "object", "properties": {}},
    risk="safe",
    category="context",
)
def front_app_info() -> ToolResult:
    out = run_applescript(_FRONT_APP_SCRIPT)
    parts = out.split(_SEP)
    app = parts[0].strip() if parts else ""
    window = parts[1].strip() if len(parts) > 1 else ""
    if not app:
        raise ToolError("I couldn't see which app is in front.")

    pairs: list[list[str]] = [["App", app]]
    if window:
        pairs.append(["Window", window])
    if app in _BROWSER_PAGE_SCRIPTS:
        try:
            page = run_applescript(_BROWSER_PAGE_SCRIPTS[app])
        except ToolError as exc:
            if str(exc) == PERMISSION_MESSAGE:
                raise
            page = ""
        page_parts = page.split(_SEP)
        if len(page_parts) >= 2:
            title, url = page_parts[0].strip(), page_parts[1].strip()
            if title:
                pairs.append(["Page", title])
            if url:
                pairs.append(["URL", url])

    if window:
        shown = window if len(window) <= 60 else window[:57].rstrip() + "…"
        speech = f"You're in {app}, on {shown}."
    else:
        speech = f"You're in {app}."
    return ToolResult(
        speech=speech,
        display={"kind": "keyvalue", "title": "Front app", "pairs": pairs},
    )


@tool(
    name="capture_screen",
    description=(
        "Take a screenshot of the whole screen and show it on the user's "
        "screen. The screen may contain sensitive information, so use this "
        "only when the user explicitly asks about their screen."
    ),
    input_schema={"type": "object", "properties": {}},
    risk="confirm",
    category="context",
    describe_call=lambda a: "Capture a screenshot of your screen",
)
def capture_screen() -> ToolResult:
    fd, path = tempfile.mkstemp(prefix="ada-screen-", suffix=".jpg")
    os.close(fd)
    try:
        try:
            proc = subprocess.run(
                ["/usr/sbin/screencapture", "-x", "-t", "jpg", path],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            raise ToolError("The screenshot timed out.") from None
        if proc.returncode != 0 or not os.path.getsize(path):
            log.debug("screencapture failed: %s", (proc.stderr or "").strip())
            raise ToolError(
                "I couldn't capture the screen — I may need Screen Recording "
                "permission in System Settings."
            )
        try:
            sips = subprocess.run(
                ["/usr/bin/sips", "--resampleWidth", "1100", path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if sips.returncode != 0:
                log.debug("sips resample failed: %s", (sips.stderr or "").strip())
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.debug("sips resample failed: %s", exc)
        with open(path, "rb") as fh:
            payload = fh.read()
    except OSError:
        raise ToolError("I couldn't capture the screen.") from None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    src = "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")
    if len(src) > _MAX_DATA_URI_CHARS:
        raise ToolError("The screenshot came out too large to display.")
    return ToolResult(
        speech="Here's your screen.",
        display={"kind": "image", "src": src, "caption": "Screenshot"},
    )
