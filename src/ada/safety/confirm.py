"""User confirmation for risky tool calls.

DialogConfirmer shows a native macOS dialog (via osascript); the dialog
gives up after 30 seconds, which counts as a denial. ConsoleConfirmer is
a terminal fallback for headless / debugging sessions.
"""

from __future__ import annotations

import logging
import subprocess

from ..tools.applescript import applescript_quote

log = logging.getLogger("ada.safety")

# The dialog auto-dismisses after this many seconds (counts as "deny").
_DIALOG_GIVE_UP_S = 30
# Hard cap on the osascript subprocess itself.
_SUBPROCESS_TIMEOUT_S = 40


class DialogConfirmer:
    """Native macOS confirmation dialog. Blocking; deny on timeout/cancel."""

    def confirm(self, title: str, message: str) -> bool:
        script = (
            f"display dialog {applescript_quote(message)} "
            f"with title {applescript_quote(title)} "
            'buttons {"Cancel", "Allow"} default button "Allow" '
            "cancel button \"Cancel\" with icon caution "
            f"giving up after {_DIALOG_GIVE_UP_S}"
        )
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            log.warning("Confirmation dialog timed out: %s", title)
            return False
        except OSError:
            log.exception("Could not launch osascript for confirmation dialog")
            return False

        # Pressing Cancel makes osascript exit nonzero.
        if result.returncode != 0:
            return False
        # The dialog "gave up" (auto-dismissed) — treat as a denial.
        if "gave up:true" in result.stdout:
            return False
        return True


class ConsoleConfirmer:
    """Terminal confirmation prompt. Anything but an explicit yes is a denial."""

    def confirm(self, title: str, message: str) -> bool:
        print(f"\n{title}")
        print(message)
        try:
            answer = input("Allow? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().lower() in ("y", "yes")
