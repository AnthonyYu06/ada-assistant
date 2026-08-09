"""Global push-to-talk hotkey (pynput).

Registers a system-wide key combination (e.g. "<cmd>+<shift>+j") that
triggers the assistant without the wake word. pynput is an optional
dependency and, on macOS, requires the Input Monitoring permission
(System Settings → Privacy & Security → Input Monitoring).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

log = logging.getLogger("ada.ui.hotkey")


def hotkey_available() -> bool:
    """True if the optional `pynput` package is importable."""
    try:
        import pynput  # noqa: F401
    except ImportError:
        return False
    return True


class HotkeyListener:
    """Listens for one global hotkey combo on a background thread.

    The combo uses pynput's GlobalHotKeys syntax, e.g. "<cmd>+<shift>+j".
    Exceptions raised by the callback are caught and logged so a buggy
    handler can never kill the OS-level key listener.
    """

    def __init__(self, combo: str, on_activate: Callable[[], None]) -> None:
        self._combo = combo
        self._on_activate = on_activate
        self._lock = threading.Lock()
        self._listener: Any | None = None  # pynput.keyboard.GlobalHotKeys

    def start(self) -> None:
        """Register the hotkey and start listening on a background thread.

        Raises:
            RuntimeError: if `pynput` is not installed.
            ValueError: if the combo string cannot be parsed.
        """
        try:
            from pynput import keyboard
        except ImportError as exc:
            raise RuntimeError(
                "The push-to-talk hotkey requires the optional 'pynput' package "
                "(pip install pynput, or: pip install 'ada-assistant[hotkey]'). "
                "On macOS, the terminal/app running Ada must also be granted "
                "Input Monitoring permission in System Settings → Privacy & Security."
            ) from exc

        with self._lock:
            if self._listener is not None:
                log.debug("HotkeyListener already started; ignoring start()")
                return
            try:
                listener = keyboard.GlobalHotKeys({self._combo: self._safe_activate})
            except ValueError as exc:
                raise ValueError(f"Invalid hotkey combo {self._combo!r}: {exc}") from exc
            listener.daemon = True
            listener.start()  # GlobalHotKeys is a thread; runs in the background
            self._listener = listener
        log.info("Global hotkey registered: %s", self._combo)

    def stop(self) -> None:
        """Stop listening. Safe to call multiple times or before start()."""
        with self._lock:
            listener, self._listener = self._listener, None
        if listener is None:
            return
        try:
            listener.stop()
        except Exception:  # noqa: BLE001 - shutdown must never raise
            log.exception("Error while stopping hotkey listener")
        log.info("Global hotkey unregistered: %s", self._combo)

    def _safe_activate(self) -> None:
        log.debug("Hotkey activated: %s", self._combo)
        try:
            self._on_activate()
        except Exception:  # noqa: BLE001 - callback bugs must not kill the listener
            log.exception("Hotkey callback raised")
