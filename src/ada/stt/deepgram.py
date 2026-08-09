"""Cloud speech-to-text via the Deepgram REST API (plain httpx, no SDK)."""

from __future__ import annotations

import io
import logging
import time
import wave

import httpx
import numpy as np

from .base import STTResult

log = logging.getLogger("ada.stt")

_LISTEN_URL = "https://api.deepgram.com/v1/listen"
_TIMEOUT_S = 20.0


class DeepgramSTT:
    """STTProvider backed by Deepgram's pre-recorded transcription endpoint."""

    name = "deepgram"

    def __init__(self, api_key: str, model: str = "nova-3") -> None:
        self.api_key = api_key
        self.model = model

    def warmup(self) -> None:
        """No-op — Deepgram is a stateless REST API."""

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> STTResult:
        """Transcribe mono int16 PCM audio via Deepgram. Blocking."""
        wav_bytes = _to_wav(audio, sample_rate)

        start = time.monotonic()
        try:
            response = httpx.post(
                _LISTEN_URL,
                params={
                    "model": self.model,
                    "smart_format": "true",
                    "language": "en",
                },
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "audio/wav",
                },
                content=wav_bytes,
                timeout=_TIMEOUT_S,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise RuntimeError("Deepgram request failed: timed out") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            reason = exc.response.reason_phrase or "HTTP error"
            raise RuntimeError(f"Deepgram request failed: {status} {reason}") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError("Deepgram request failed: connection error") from exc
        latency = time.monotonic() - start

        try:
            data = response.json()
            text = data["results"]["channels"][0]["alternatives"][0]["transcript"]
        except (ValueError, KeyError, IndexError, TypeError):
            log.warning("Unexpected Deepgram response shape; returning empty transcript")
            text = ""

        result = STTResult(
            text=(text or "").strip(),
            provider=self.name,
            audio_s=len(audio) / sample_rate,
            latency_s=latency,
        )
        log.debug(
            "Deepgram transcribed %.2fs of audio in %.2fs: %r",
            result.audio_s,
            result.latency_s,
            result.text,
        )
        return result


def _to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono int16 PCM samples as an in-memory WAV file."""
    pcm = np.ascontiguousarray(audio, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()
