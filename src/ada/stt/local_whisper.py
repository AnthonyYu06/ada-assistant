"""Local speech-to-text via faster-whisper (CTranslate2 Whisper).

The model is loaded lazily on first use (or via warmup()) so importing
this module stays cheap and startup can defer the expensive load.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from .base import STTResult

log = logging.getLogger("ada.stt")

_EXPECTED_SAMPLE_RATE = 16000


class LocalWhisperSTT:
    """STTProvider backed by a local faster-whisper model on CPU."""

    name = "local-whisper"

    def __init__(self, model_name: str = "small.en", compute_type: str = "int8") -> None:
        self.model_name = model_name
        self.compute_type = compute_type
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        """Load the Whisper model (thread-safe, idempotent)."""
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                log.info(
                    "Loading faster-whisper model %r (compute_type=%s)...",
                    self.model_name,
                    self.compute_type,
                )
                start = time.monotonic()
                self._model = WhisperModel(
                    self.model_name, device="cpu", compute_type=self.compute_type
                )
                log.info(
                    "Whisper model %r loaded in %.1fs",
                    self.model_name,
                    time.monotonic() - start,
                )
            return self._model

    def warmup(self) -> None:
        """Load the model and prime the compute kernels with 0.5 s of silence."""
        model = self._load()
        silence = np.zeros(_EXPECTED_SAMPLE_RATE // 2, dtype=np.float32)
        start = time.monotonic()
        segments, _info = model.transcribe(
            silence,
            language=self._language(),
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        # The generator is lazy — drain it so the kernels actually run.
        for _segment in segments:
            pass
        log.debug("Whisper warmup transcription took %.2fs", time.monotonic() - start)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> STTResult:
        """Transcribe mono int16 PCM audio at 16 kHz. Blocking."""
        if sample_rate != _EXPECTED_SAMPLE_RATE:
            raise ValueError(
                f"LocalWhisperSTT requires {_EXPECTED_SAMPLE_RATE} Hz audio, "
                f"got {sample_rate} Hz"
            )
        model = self._load()
        audio_f32 = audio.astype(np.float32) / 32768.0

        start = time.monotonic()
        segments, _info = model.transcribe(
            audio_f32,
            language=self._language(),
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        text = "".join(segment.text for segment in segments).strip()
        latency = time.monotonic() - start

        result = STTResult(
            text=text,
            provider=self.name,
            audio_s=len(audio) / sample_rate,
            latency_s=latency,
        )
        log.debug(
            "Transcribed %.2fs of audio in %.2fs: %r",
            result.audio_s,
            result.latency_s,
            text,
        )
        return result

    def _language(self) -> str | None:
        return "en" if self.model_name.endswith(".en") else None
