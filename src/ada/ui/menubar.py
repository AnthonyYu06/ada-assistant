"""macOS menu-bar status app for Ada (rumps).

The menu bar shows the current status icon and offers a status line, a
"Mute microphone" toggle, and a Quit item. The assistant main loop runs
on a worker thread; rumps (and the AppKit event loop underneath it) must
own the process main thread.

rumps is an optional dependency — install with `pip install rumps` or
`pip install 'ada-assistant[menubar]'`.
"""

from __future__ import annotations

import logging
import threading

from ..state import STATUS_ICONS, AssistantState, Status

log = logging.getLogger("ada.ui.menubar")

_MAX_DETAIL_CHARS = 60


def menubar_available() -> bool:
    """True if the optional `rumps` package is importable."""
    try:
        import rumps  # noqa: F401
    except ImportError:
        return False
    return True


def run_menubar(state: AssistantState, assistant_name: str, worker: threading.Thread) -> None:
    """Run the menu-bar app on the current (main) thread.

    Starts `worker` (the assistant main loop) first, then enters
    ``rumps.App.run()``, which BLOCKS the main thread inside the macOS
    event loop until the app quits. This is expected: call this from the
    process main thread and put everything else on the worker thread.

    A 0.5 s rumps.Timer polls the shared state to keep the title icon,
    the status line, and the mute checkmark in sync, and it quits the
    app once ``state.shutting_down`` is set (whether by the Quit menu
    item or by any other component requesting shutdown).

    Raises:
        RuntimeError: if `rumps` is not installed.
    """
    try:
        import rumps
    except ImportError as exc:
        raise RuntimeError(
            "The menu-bar UI requires the optional 'rumps' package. "
            "Install it with: pip install rumps  "
            "(or: pip install 'ada-assistant[menubar]')"
        ) from exc

    app = rumps.App(assistant_name, title=STATUS_ICONS[Status.STARTING], quit_button=None)

    # A MenuItem without a callback renders disabled (greyed out) — it is
    # our read-only status line.
    status_item = rumps.MenuItem("Starting…")

    def on_mute(item: "rumps.MenuItem") -> None:
        muted = state.toggle_muted()
        item.state = 1 if muted else 0
        log.info("Microphone %s from menu bar", "muted" if muted else "unmuted")

    mute_item = rumps.MenuItem("Mute microphone", callback=on_mute)

    def on_quit(_item: "rumps.MenuItem") -> None:
        log.info("Quit requested from menu bar")
        state.request_shutdown()
        # Give the worker a moment to wind down; the timer below performs
        # the actual rumps.quit_application() once shutting_down is seen.
        if worker.is_alive():
            worker.join(timeout=2.0)

    quit_item = rumps.MenuItem(f"Quit {assistant_name}", callback=on_quit)

    app.menu = [status_item, None, mute_item, None, quit_item]

    last = {"icon": "", "line": "", "muted": None}

    def tick(_timer: "rumps.Timer") -> None:
        if state.shutting_down:
            timer.stop()
            if worker.is_alive():
                worker.join(timeout=2.0)
            rumps.quit_application()
            return
        status = state.status
        detail = state.detail
        icon = STATUS_ICONS.get(status, "?")
        line = status.value.capitalize()
        if detail:
            trimmed = detail if len(detail) <= _MAX_DETAIL_CHARS else detail[: _MAX_DETAIL_CHARS - 1] + "…"
            line += f" — {trimmed}"
        muted = state.muted
        if icon != last["icon"]:
            app.title = icon
            last["icon"] = icon
        if line != last["line"]:
            status_item.title = line
            last["line"] = line
        if muted != last["muted"]:
            mute_item.state = 1 if muted else 0
            last["muted"] = muted

    timer = rumps.Timer(tick, 0.5)
    timer.start()

    # The worker is the assistant main loop; it must be running before we
    # block in app.run(). Tolerate a caller that already started it.
    if worker.ident is None:
        worker.start()

    log.info("Entering menu-bar event loop (main thread blocks here)")
    app.run()
