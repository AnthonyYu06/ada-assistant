"""ElevenLabs TTS provider.

Streams raw PCM from the ElevenLabs REST API (no SDK) and plays it through
sounddevice as it arrives, so speech starts before synthesis finishes.
Playback is interruptible from another thread via stop().
"""

from __future__ import annotations

import logging
import threading

import httpx
import sounddevice as sd

log = logging.getLogger("ada.tts.elevenlabs")

_SAMPLE_RATE = 22050
_CHUNK_SIZE = 4096
_BYTES_PER_SAMPLE = 2  # int16 mono


class ElevenLabsTTS:
    """TTSProvider backed by the ElevenLabs streaming REST API."""

    name = "elevenlabs"

    def __init__(self, api_key: str, voice_id: str, model: str = "eleven_flash_v2_5") -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._interrupt = threading.Event()
        self._lock = threading.Lock()
        self._stream: sd.RawOutputStream | None = None

    # -- TTSProvider ------------------------------------------------------
    def arm(self) -> None:
        """Clear the interrupt flag so the next speak() streams audio."""
        with self._lock:
            self._interrupt.clear()

    def speak(self, text: str) -> None:
        """Synthesize and play `text`, blocking until done or stop() is called."""
        if not text or not text.strip():
            return

        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream"
            f"?output_format=pcm_{_SAMPLE_RATE}"
        )
        headers = {"xi-api-key": self._api_key, "Content-Type": "application/json"}
        payload = {"text": text, "model_id": self._model}

        out: sd.RawOutputStream | None = None
        try:
            with httpx.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=httpx.Timeout(30.0, connect=10.0),
            ) as resp:
                if resp.status_code != 200:
                    snippet = resp.read()[:200].decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"ElevenLabs request failed: HTTP {resp.status_code}: {snippet}"
                    )

                out = sd.RawOutputStream(
                    samplerate=_SAMPLE_RATE, channels=1, dtype="int16"
                )
                with self._lock:
                    self._stream = out
                out.start()

                carry = b""  # holds a trailing odd byte between chunks
                for chunk in resp.iter_bytes(chunk_size=_CHUNK_SIZE):
                    if self._interrupt.is_set():
                        break
                    data = carry + chunk
                    cut = len(data) - (len(data) % _BYTES_PER_SAMPLE)
                    carry = data[cut:]
                    if not cut:
                        continue
                    try:
                        out.write(data[:cut])
                    except sd.PortAudioError:
                        if self._interrupt.is_set():
                            break  # stop() aborted the stream under our feet
                        raise
        finally:
            with self._lock:
                self._stream = None
            if out is not None:
                try:
                    if self._interrupt.is_set():
                        out.abort()  # discard anything still buffered
                    else:
                        out.stop()  # drain buffered audio to the speaker
                except Exception as exc:  # noqa: BLE001 - stream may already be closed by stop()
                    log.debug("Error finalizing output stream: %s", exc)
                try:
                    out.close()
                except Exception as exc:  # noqa: BLE001
                    log.debug("Error closing output stream: %s", exc)

    def stop(self) -> None:
        """Interrupt any in-progress playback. Safe from any thread."""
        self._interrupt.set()
        with self._lock:
            stream = self._stream
            if stream is not None:
                try:
                    stream.abort()
                    stream.close()
                except Exception as exc:  # noqa: BLE001 - racing with speak() teardown
                    log.debug("Error aborting output stream: %s", exc)
