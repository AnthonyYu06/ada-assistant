"""Microphone capture.

MicStream wraps a sounddevice.InputStream and hands out fixed-size mono
int16 frames through a bounded queue. The PortAudio callback runs on a
high-priority audio thread; consumers (wake word / VAD / STT) pull frames
with read() from ordinary Python threads.
"""

from __future__ import annotations

import logging
import queue
import time
from types import TracebackType

import numpy as np
import sounddevice as sd

from ..config import Config

log = logging.getLogger("ada.audio.capture")

# Bounded so a stalled consumer can never grow memory without limit.
_QUEUE_MAX_FRAMES = 200


class MicStream:
    """Continuous microphone capture producing 1-D int16 frames.

    Frames are exactly ``cfg.audio.frame_samples`` samples long (80 ms at
    16 kHz by default — the size openWakeWord expects). If the consumer
    falls behind, the oldest frames are dropped.
    """

    def __init__(self, cfg: Config) -> None:
        self.sample_rate: int = cfg.audio.sample_rate
        self.frame_samples: int = cfg.audio.frame_samples
        self.device: int | str | None = cfg.audio.input_device
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=_QUEUE_MAX_FRAMES)
        self._stream: sd.InputStream | None = None
        # Rate-limiters for log noise from the audio callback.
        self._last_drop_warning = 0.0
        self._last_status_log = 0.0

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        """Open and start the input stream. Safe to call if already running."""
        if self._stream is not None:
            return
        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.frame_samples,
                channels=1,
                dtype="int16",
                device=self.device,
                callback=self._callback,
            )
            stream.start()
        except Exception as exc:  # PortAudioError, ValueError (bad device), ...
            device_hint = (
                f" (input_device={self.device!r})" if self.device is not None else ""
            )
            raise RuntimeError(
                f"Could not open the microphone{device_hint}: {exc}. "
                "Make sure a microphone is connected and that this app "
                "(your terminal, when running from the command line) is allowed "
                "to use it under System Settings > Privacy & Security > Microphone. "
                "Run `ada doctor` to list available input devices."
            ) from exc
        self._stream = stream
        log.info(
            "Microphone stream started (device=%s, %d Hz, %d-sample frames)",
            "default" if self.device is None else repr(self.device),
            self.sample_rate,
            self.frame_samples,
        )

    def stop(self) -> None:
        """Stop and close the input stream. Safe to call when not running."""
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            log.debug("Error while closing input stream: %s", exc)
        log.info("Microphone stream stopped")

    @property
    def running(self) -> bool:
        return self._stream is not None

    def __enter__(self) -> MicStream:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    # -- frame access -----------------------------------------------------
    def read(self, timeout: float | None = 0.5) -> np.ndarray | None:
        """Return the next captured frame, or None if `timeout` elapses.

        A timeout of None blocks until a frame arrives.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self) -> None:
        """Drop any frames currently buffered (e.g. audio captured while
        the assistant was speaking)."""
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    # -- PortAudio callback (audio thread — keep fast, never raise) --------
    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            # Input overflow etc. — expected under load; keep it quiet.
            now = time.monotonic()
            if now - self._last_status_log >= 1.0:
                self._last_status_log = now
                log.debug("Input stream status: %s", status)
        frame = indata[:, 0].copy()  # PortAudio reuses the buffer — must copy
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            # Drop the oldest frame to make room; warn at most once a second.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass
            now = time.monotonic()
            if now - self._last_drop_warning >= 1.0:
                self._last_drop_warning = now
                log.warning(
                    "Audio frame queue full — dropping oldest audio "
                    "(consumer is falling behind)"
                )
