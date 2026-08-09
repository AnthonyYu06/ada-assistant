"""Voice-activity-based utterance endpointing (Silero VAD).

UtteranceRecorder pulls frames from a MicStream after the wake word fires
and returns one complete utterance: pre-roll audio from just before speech
began, the speech itself, and the trailing silence that ended it.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque

import numpy as np
from pysilero_vad import SileroVoiceActivityDetector

from ..config import Config
from .capture import MicStream

log = logging.getLogger("ada.audio.vad")


class UtteranceRecorder:
    """Records a single utterance using Silero VAD for endpointing."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg.vad
        self._sample_rate = cfg.audio.sample_rate
        self._frame_samples = cfg.audio.frame_samples
        self._detector = SileroVoiceActivityDetector()
        # Silero processes fixed-size chunks (512 samples @ 16 kHz).
        self._chunk_samples = int(self._detector.chunk_samples())

    @property
    def chunk_samples(self) -> int:
        """Samples per VAD chunk (used by doctor to feed is_speech())."""
        return self._chunk_samples

    def is_speech(self, chunk_bytes: bytes) -> float:
        """Raw Silero speech probability for one chunk of int16 PCM bytes."""
        return float(self._detector(chunk_bytes))

    def _reset_detector(self) -> None:
        for name in ("reset", "reset_states"):
            method = getattr(self._detector, name, None)
            if callable(method):
                try:
                    method()
                except Exception as exc:  # noqa: BLE001 - reset is best-effort
                    log.debug("VAD %s() failed: %s", name, exc)
                return

    def record(
        self, mic: MicStream, cancel: threading.Event | None = None
    ) -> np.ndarray | None:
        """Record one utterance from `mic`.

        Returns the utterance (pre-roll + speech + trailing silence) as a
        1-D int16 array, or None if speech never started, the recording was
        cancelled, or nothing usable was captured.
        """
        cfg = self._cfg
        sample_rate = self._sample_rate
        self._reset_detector()

        # Pre-roll: recent frames kept from just before speech starts.
        frame_ms_nominal = self._frame_samples * 1000.0 / sample_rate
        preroll_frames = max(1, math.ceil(cfg.pre_roll_ms / frame_ms_nominal))
        preroll: deque[np.ndarray] = deque(maxlen=preroll_frames)

        pending: list[np.ndarray] = []   # frames in the current speech run-up
        recorded: list[np.ndarray] = []  # the utterance, once speech confirmed
        carry = np.empty(0, dtype=np.int16)  # partial VAD chunk between frames

        started = False
        speech_run_ms = 0.0
        trailing_silence_ms = 0.0
        recorded_ms = 0.0
        last_prob = 0.0

        start_time = time.monotonic()
        # Absolute safety net so a stalled mic can never wedge this loop.
        hard_deadline = (
            start_time
            + cfg.no_speech_timeout_s
            + cfg.max_utterance_s
            + cfg.end_silence_ms / 1000.0
            + 5.0
        )

        while True:
            if cancel is not None and cancel.is_set():
                log.debug("Utterance recording cancelled")
                return None
            now = time.monotonic()
            # Only abort on the no-speech timeout when nothing is actually
            # accumulating. If a speech run-up is underway (buffered in
            # `pending`) but has not yet crossed min_speech_ms, keep going so
            # a slow-to-start utterance is not clipped away.
            if (
                not started
                and speech_run_ms == 0.0
                and not pending
                and now - start_time >= cfg.no_speech_timeout_s
            ):
                log.debug(
                    "No speech within %.1fs — giving up", cfg.no_speech_timeout_s
                )
                return None
            if now >= hard_deadline:
                log.warning("Utterance recording hit its hard time limit")
                break

            frame = mic.read(timeout=0.5)
            if frame is None:
                continue
            frame = np.asarray(frame, dtype=np.int16).reshape(-1)

            # Re-chunk mic frames (1280 samples) into VAD chunks (512 samples),
            # carrying the remainder over to the next frame.
            carry = np.concatenate((carry, frame)) if carry.size else frame
            probs: list[float] = []
            while carry.size >= self._chunk_samples:
                chunk = carry[: self._chunk_samples]
                carry = carry[self._chunk_samples :]
                probs.append(float(self._detector(chunk.tobytes())))
            if probs:
                last_prob = probs[-1]
                frame_has_speech = max(probs) >= cfg.threshold
            else:
                # No full chunk completed in this frame; reuse the last verdict.
                frame_has_speech = last_prob >= cfg.threshold

            frame_ms = frame.size * 1000.0 / sample_rate

            if not started:
                if frame_has_speech:
                    pending.append(frame)
                    speech_run_ms += frame_ms
                    if speech_run_ms >= cfg.min_speech_ms:
                        started = True
                        recorded = list(preroll) + pending
                        pending = []
                        recorded_ms = (
                            sum(f.size for f in recorded) * 1000.0 / sample_rate
                        )
                        trailing_silence_ms = 0.0
                        log.debug(
                            "Speech started (%.0f ms pre-roll kept)",
                            min(cfg.pre_roll_ms, len(preroll) * frame_ms_nominal),
                        )
                else:
                    # The blip was shorter than min_speech_ms — fold it back
                    # into the pre-roll so nothing is lost if speech starts soon.
                    speech_run_ms = 0.0
                    preroll.extend(pending)
                    pending = []
                    preroll.append(frame)
            else:
                recorded.append(frame)
                recorded_ms += frame_ms
                if frame_has_speech:
                    trailing_silence_ms = 0.0
                else:
                    trailing_silence_ms += frame_ms
                if trailing_silence_ms >= cfg.end_silence_ms:
                    log.debug(
                        "Utterance ended after %.0f ms of silence (%.1f s total)",
                        trailing_silence_ms,
                        recorded_ms / 1000.0,
                    )
                    break
                if recorded_ms >= cfg.max_utterance_s * 1000.0:
                    log.debug(
                        "Utterance hit the %.1f s cap", cfg.max_utterance_s
                    )
                    break

        if not recorded:
            return None
        return np.concatenate(recorded)
