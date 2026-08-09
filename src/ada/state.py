"""Shared, thread-safe assistant state.

The audio pipeline, the brain, and the UI all run on different threads.
This module is the single place they exchange status.
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Callable


class Status(str, Enum):
    STARTING = "starting"
    IDLE = "idle"           # waiting for the wake word
    LISTENING = "listening" # recording an utterance
    THINKING = "thinking"   # transcribing / calling the model / running tools
    SPEAKING = "speaking"   # playing a response
    MUTED = "muted"         # mic input is being discarded
    ERROR = "error"
    STOPPED = "stopped"


# Emoji used by the menu bar / console for each status.
STATUS_ICONS: dict[Status, str] = {
    Status.STARTING: "…",
    Status.IDLE: "😴",
    Status.LISTENING: "🎙️",
    Status.THINKING: "🤔",
    Status.SPEAKING: "🗣️",
    Status.MUTED: "🔇",
    Status.ERROR: "⚠️",
    Status.STOPPED: "⏹",
}

StatusListener = Callable[[Status, str], None]


class AssistantState:
    """Thread-safe status + mute + shutdown flags shared across components."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = Status.STARTING
        self._detail = ""
        self._muted = False
        self._listeners: list[StatusListener] = []
        self.shutdown_event = threading.Event()
        self.last_activity = time.monotonic()

    # -- status ---------------------------------------------------------
    @property
    def status(self) -> Status:
        with self._lock:
            return self._status

    @property
    def detail(self) -> str:
        with self._lock:
            return self._detail

    def set_status(self, status: Status, detail: str = "") -> None:
        with self._lock:
            self._status = status
            self._detail = detail
            listeners = list(self._listeners)
        self.last_activity = time.monotonic()
        for listener in listeners:
            try:
                listener(status, detail)
            except Exception:  # noqa: BLE001 - UI failures must not kill the pipeline
                pass

    def set_status_if(
        self, expected: Status, new: Status, detail: str = ""
    ) -> bool:
        """Atomically set status to `new` only if it is currently `expected`.

        Does nothing while shutting down. Returns True if the swap happened.
        Listeners fire outside the lock, exactly like set_status.
        """
        with self._lock:
            if self.shutting_down or self._status != expected:
                return False
            self._status = new
            self._detail = detail
            listeners = list(self._listeners)
        self.last_activity = time.monotonic()
        for listener in listeners:
            try:
                listener(new, detail)
            except Exception:  # noqa: BLE001 - UI failures must not kill the pipeline
                pass
        return True

    def add_listener(self, listener: StatusListener) -> None:
        with self._lock:
            self._listeners.append(listener)

    # -- mute -----------------------------------------------------------
    @property
    def muted(self) -> bool:
        with self._lock:
            return self._muted

    def set_muted(self, muted: bool) -> None:
        """Flip the mute flag; only rewrite status from a resting state.

        The status doubles as mic-queue ownership, so an in-flight turn's
        LISTENING/THINKING/SPEAKING/ERROR status must never be stomped here.
        Only a resting IDLE/MUTED status is rewritten to reflect the new mute.
        """
        with self._lock:
            self._muted = muted
            resting = self._status in (Status.IDLE, Status.MUTED)
            if resting:
                self._status = Status.MUTED if muted else Status.IDLE
                new_status = self._status
                listeners = list(self._listeners)
        if not resting:
            return
        self.last_activity = time.monotonic()
        for listener in listeners:
            try:
                listener(new_status, "")
            except Exception:  # noqa: BLE001 - UI failures must not kill the pipeline
                pass

    def toggle_muted(self) -> bool:
        self.set_muted(not self.muted)
        return self.muted

    # -- shutdown (kill switch) ------------------------------------------
    def request_shutdown(self) -> None:
        self.shutdown_event.set()
        self.set_status(Status.STOPPED)

    @property
    def shutting_down(self) -> bool:
        return self.shutdown_event.is_set()
