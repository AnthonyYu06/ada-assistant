"""Text-to-speech provider contract."""

from __future__ import annotations

from typing import Protocol


class TTSProvider(Protocol):
    name: str

    def arm(self) -> None:
        """Reset the per-utterance interrupt state (clear the interrupt Event).

        Thread-safe and idempotent. Called by the Speaker worker under the
        Speaker lock right before it commits to speaking a live chunk, so that
        speak() itself no longer clears the interrupt (it only reads it)."""
        ...

    def speak(self, text: str) -> None:
        """Speak one chunk of text. Blocking until playback finishes
        or stop() is called from another thread."""
        ...

    def stop(self) -> None:
        """Immediately interrupt any in-progress playback. Thread-safe."""
        ...
