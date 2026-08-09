"""Speech-to-text provider contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class STTResult:
    text: str
    provider: str
    audio_s: float = 0.0    # duration of the audio transcribed
    latency_s: float = 0.0  # wall-clock transcription time


class STTProvider(Protocol):
    name: str

    def warmup(self) -> None:
        """Load models / open connections so the first request is fast."""
        ...

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> STTResult:
        """Transcribe mono int16 PCM audio (1-D numpy array). Blocking."""
        ...
