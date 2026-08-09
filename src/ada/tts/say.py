"""macOS `say` TTS provider.

Shells out to /usr/bin/say, which plays through the default output device.
Playback is interruptible from another thread via stop().
"""

from __future__ import annotations

import logging
import subprocess
import threading

log = logging.getLogger("ada.tts.say")

_SAY_BIN = "/usr/bin/say"


class SayTTS:
    """TTSProvider backed by the built-in macOS `say` command."""

    name = "say"

    def __init__(self, voice: str = "Samantha", rate_wpm: int = 190) -> None:
        self._voice = voice
        self._rate_wpm = int(rate_wpm)
        self._lock = threading.Lock()
        self._interrupt = threading.Event()
        self._proc: subprocess.Popen[bytes] | None = None
        # If the configured voice turns out to be unknown, we drop -v and
        # remember that for all subsequent calls.
        self._omit_voice = not voice
        self._used = False

    # -- TTSProvider ------------------------------------------------------
    def arm(self) -> None:
        """Clear the interrupt flag so the next speak() may launch."""
        with self._lock:
            self._interrupt.clear()

    def speak(self, text: str) -> None:
        """Speak `text`, blocking until playback finishes or stop() is called."""
        if not text or not text.strip():
            return
        if self._interrupt.is_set():
            return  # stop() landed before we committed to speaking
        first_use = not self._used
        self._used = True

        returncode = self._run(text, omit_voice=self._omit_voice)
        # A positive exit status on the very first use usually means the
        # configured voice does not exist. (Negative = killed by stop().)
        if returncode > 0 and first_use and not self._omit_voice:
            log.warning(
                "'say' exited with status %d using voice %r; "
                "retrying without -v and omitting it from now on",
                returncode,
                self._voice,
            )
            self._omit_voice = True
            self._run(text, omit_voice=True)

    def stop(self) -> None:
        """Interrupt any in-progress playback. Safe from any thread."""
        with self._lock:
            self._interrupt.set()
            proc = self._proc
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except (ProcessLookupError, OSError) as exc:
                    log.debug("Could not terminate 'say' process: %s", exc)

    # -- internals --------------------------------------------------------
    def _run(self, text: str, omit_voice: bool) -> int:
        cmd = [_SAY_BIN]
        if not omit_voice:
            cmd += ["-v", self._voice]
        cmd += ["-r", str(self._rate_wpm), "--", text]

        with self._lock:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._proc = proc
            if self._interrupt.is_set():
                # stop() fired between our speak()-entry check and registering
                # the process; terminate it now so the sentence does not play.
                try:
                    proc.terminate()
                except (ProcessLookupError, OSError) as exc:
                    log.debug("Could not terminate 'say' process: %s", exc)
        try:
            return proc.wait()
        finally:
            with self._lock:
                if self._proc is proc:
                    self._proc = None
