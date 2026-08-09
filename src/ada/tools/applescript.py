"""Safe AppleScript execution helpers.

Every string interpolated into an AppleScript program MUST go through
:func:`applescript_quote` — never f-string raw user input into a script.
"""

from __future__ import annotations

import logging
import re
import subprocess
from datetime import datetime

from .registry import ToolError

log = logging.getLogger("ada.tools.applescript")

# Speakable message used whenever macOS refuses because the host process
# lacks Automation / Accessibility permission.
PERMISSION_MESSAGE = (
    "I need Automation or Accessibility permission for that — "
    "grant it in System Settings, Privacy and Security."
)

# All markers are lowercase; the check compares against lowercased stderr.
_PERMISSION_MARKERS = (
    "not allowed assistive access",
    "not authorized to send apple events",
    "-25211",
    "-1719",
    "-1743",
)


def applescript_quote(s: str) -> str:
    """Return `s` as a double-quoted AppleScript string literal."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def run_applescript(script: str, timeout: float = 10.0) -> str:
    """Run an AppleScript program via osascript and return stripped stdout.

    Raises ToolError (speakable) on timeout, permission problems, or any
    nonzero exit.
    """
    try:
        proc = subprocess.run(
            ["/usr/bin/osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise ToolError("That system action timed out.") from None
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        log.debug("osascript exited %d: %s", proc.returncode, stderr)
        # Permission check runs on the raw stderr, before any regex strips the
        # trailing "(-1743)" code that we key on.
        lowered = stderr.lower()
        if any(marker in lowered for marker in _PERMISSION_MARKERS):
            raise ToolError(PERMISSION_MESSAGE)
        line = stderr.splitlines()[0].strip() if stderr else "unknown AppleScript error"
        # Drop osascript's "123:456: execution error:" prefix and trailing codes.
        line = re.sub(r"^\d+:\d+:\s*(execution error:\s*)?", "", line)
        line = re.sub(r"\s*\(-?\d+\)\s*$", "", line).strip() or "unknown AppleScript error"
        raise ToolError(f"That didn't work: {line}")
    return proc.stdout.strip()


def applescript_date(iso: str) -> str:
    """Build a locale-independent AppleScript snippet defining `theDate`.

    Accepts ISO "YYYY-MM-DDTHH:MM" (seconds optional). The snippet sets
    day to 1 first so switching months can never overflow.
    """
    try:
        dt = datetime.fromisoformat(str(iso).strip())
    except (TypeError, ValueError):
        raise ToolError(
            f"I couldn't understand the date {iso!r} — use the format 2026-07-13T15:42."
        ) from None
    return "\n".join(
        [
            "set theDate to current date",
            "set day of theDate to 1",
            f"set year of theDate to {dt.year}",
            f"set month of theDate to {dt.month}",
            f"set day of theDate to {dt.day}",
            f"set time of theDate to ({dt.hour} * hours + {dt.minute} * minutes + {dt.second})",
        ]
    )
