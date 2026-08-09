"""Audible cues via macOS system sounds.

Each chime spawns `afplay` non-blocking and silently no-ops when disabled,
when the sound file is missing, or when afplay is unavailable.
"""

from __future__ import annotations

import logging
import os
import subprocess

log = logging.getLogger("ada.audio.chime")

_AFPLAY = "/usr/bin/afplay"

_WAKE_SOUND = "/System/Library/Sounds/Pop.aiff"
_DONE_SOUND = "/System/Library/Sounds/Bottle.aiff"
_ERROR_SOUND = "/System/Library/Sounds/Basso.aiff"
_TIMER_SOUND = "/System/Library/Sounds/Glass.aiff"


def _play(path: str, enabled: bool) -> None:
    if not enabled:
        return
    if not os.path.isfile(path) or not os.path.isfile(_AFPLAY):
        return
    try:
        subprocess.Popen(
            [_AFPLAY, path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        log.debug("Could not play chime %s: %s", path, exc)


def chime_wake(enabled: bool = True) -> None:
    """Wake word heard — the assistant is listening."""
    _play(_WAKE_SOUND, enabled)


def chime_done(enabled: bool = True) -> None:
    """Finished listening / request handled."""
    _play(_DONE_SOUND, enabled)


def chime_error(enabled: bool = True) -> None:
    """Something went wrong."""
    _play(_ERROR_SOUND, enabled)


def chime_timer(enabled: bool = True) -> None:
    """A timer or reminder fired."""
    _play(_TIMER_SOUND, enabled)
