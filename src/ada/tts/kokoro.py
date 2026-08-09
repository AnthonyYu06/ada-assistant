"""Kokoro TTS provider — a natural, local, free neural voice.

Kokoro is a small (82M-param) neural text-to-speech model that runs on-device
via onnxruntime (no PyTorch, no cloud, no API key). It sounds dramatically
more natural than the built-in macOS ``say`` voice — the closest free, local
thing to a service like ElevenLabs.

The model weights (~310 MB) live in a cache dir and are downloaded once by
``ada setup`` (or lazily on first use). Synthesis and playback are split into
``synthesize()`` / ``play_prepared()`` (the Speaker's optional pipelining
capability), so the Speaker can synthesize the next sentence on its prefetch
thread while the current one plays; ``speak()`` composes the two for serial
callers. Playback streams to the audio device in small chunks so a barge-in
(stop()) interrupts it immediately.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import httpx
import numpy as np
import sounddevice as sd

log = logging.getLogger("ada.tts.kokoro")

_SAMPLE_RATE = 24000  # Kokoro output rate
_MODEL_FILE = "kokoro-v1.0.onnx"
_VOICES_FILE = "voices-v1.0.bin"
_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
_MODEL_URL = f"{_BASE}/{_MODEL_FILE}"
_VOICES_URL = f"{_BASE}/{_VOICES_FILE}"


def model_dir() -> Path:
    return Path.home() / ".cache" / "ada" / "kokoro"


def model_paths() -> tuple[Path, Path]:
    d = model_dir()
    return d / _MODEL_FILE, d / _VOICES_FILE


def models_present() -> bool:
    model, voices = model_paths()
    # A truncated/failed download leaves a tiny file; require plausible sizes.
    return (
        model.is_file() and model.stat().st_size > 50_000_000
        and voices.is_file() and voices.stat().st_size > 1_000_000
    )


def ensure_models(progress: "callable | None" = None) -> None:
    """Download the Kokoro model + voices if missing. Raises on failure."""
    if models_present():
        return
    d = model_dir()
    d.mkdir(parents=True, exist_ok=True)
    for url, name in ((_MODEL_URL, _MODEL_FILE), (_VOICES_URL, _VOICES_FILE)):
        dest = d / name
        if dest.is_file() and dest.stat().st_size > 1_000_000:
            continue
        if progress:
            progress(f"Downloading {name}…")
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
        except Exception as exc:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download the Kokoro voice model ({name}): {exc}"
            ) from exc


def _lang_for(voice: str, override: str | None) -> str:
    if override:
        return override
    return "en-gb" if voice[:1] == "b" else "en-us"


class KokoroTTS:
    """TTSProvider backed by the local Kokoro neural model (onnxruntime)."""

    name = "kokoro"

    def __init__(
        self,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang: str | None = None,
    ) -> None:
        self._voice = voice
        self._speed = float(speed)
        self._lang = _lang_for(voice, lang)
        self._model: object | None = None
        self._load_lock = threading.Lock()
        self._interrupt = threading.Event()
        self._lock = threading.Lock()
        self._stream: sd.RawOutputStream | None = None
        # Warm the model in the background so the first reply isn't delayed.
        threading.Thread(target=self._preload, name="kokoro-preload", daemon=True).start()

    # -- model loading ------------------------------------------------------
    def _load(self) -> object:
        with self._load_lock:
            if self._model is None:
                from kokoro_onnx import Kokoro

                model, voices = model_paths()
                if not models_present():
                    raise RuntimeError(
                        "Kokoro voice model is not downloaded — run `ada setup`."
                    )
                self._model = Kokoro(str(model), str(voices))
        return self._model

    def _preload(self) -> None:
        try:
            self._load()
            log.info("Kokoro voice %r ready (local neural TTS).", self._voice)
        except Exception as exc:  # noqa: BLE001 - non-fatal; surfaced on first speak
            log.warning("Kokoro preload deferred: %s", exc)

    def warmup(self) -> None:
        try:
            self._synthesize("Ready.")
        except Exception:  # noqa: BLE001
            log.exception("Kokoro warmup failed")

    # -- TTSProvider --------------------------------------------------------
    def arm(self) -> None:
        with self._lock:
            self._interrupt.clear()

    def speak(self, text: str) -> None:
        """Serial path: synthesize then play. Kept for callers/providers that
        do not use the Speaker's pipelining capability (e.g. warm-up, `ada
        say`, tests) — it simply composes synthesize() + play_prepared()."""
        try:
            pcm = self.synthesize(text)
        except Exception:  # noqa: BLE001 - a synth failure must not kill the worker
            log.exception("Kokoro synthesis failed")
            return
        self.play_prepared(pcm)

    # -- Speaker pipelining capability (synthesize / play_prepared) ----------
    def synthesize(self, text: str) -> np.ndarray | None:
        """Model inference only — no audio I/O, safe to run on the Speaker's
        prefetch thread while another chunk is playing (the Speaker never
        runs two syntheses concurrently). Returns an opaque prepared-audio
        object for play_prepared(), or None for blank/empty output. Raises on
        synthesis failure (the Speaker logs and skips the chunk)."""
        if not text or not text.strip():
            return None
        return self._synthesize(text)

    def play_prepared(self, prepared: np.ndarray | None) -> None:
        """Play a prepared object from synthesize(). Same interrupt contract
        as speak(): a stop() that already landed (interrupt set) skips
        playback entirely, and a stop() during playback aborts within one
        ~100 ms chunk. The Speaker decides *whether* to call this at all via
        its play-time generation check; the interrupt checks here are the
        provider-level half of that barge-in contract."""
        if prepared is None or self._interrupt.is_set():
            return
        self._play(prepared)

    def stop(self) -> None:
        self._interrupt.set()
        with self._lock:
            stream = self._stream
            if stream is not None:
                try:
                    stream.abort()
                    stream.close()
                except Exception as exc:  # noqa: BLE001 - racing with speak() teardown
                    log.debug("Error aborting Kokoro stream: %s", exc)

    # -- internals ----------------------------------------------------------
    def _synthesize(self, text: str) -> np.ndarray | None:
        model = self._load()
        samples, _sr = model.create(  # type: ignore[attr-defined]
            text, voice=self._voice, speed=self._speed, lang=self._lang
        )
        arr = np.asarray(samples, dtype=np.float32)
        if arr.size == 0:
            return None
        arr = np.clip(arr, -1.0, 1.0)
        return (arr * 32767.0).astype(np.int16)

    def _play(self, pcm: np.ndarray) -> None:
        out = sd.RawOutputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="int16")
        with self._lock:
            self._stream = out
        try:
            out.start()
            data = pcm.tobytes()
            step = _SAMPLE_RATE // 10 * 2  # ~100 ms of int16 mono
            for i in range(0, len(data), step):
                if self._interrupt.is_set():
                    break
                try:
                    out.write(data[i : i + step])
                except sd.PortAudioError:
                    if self._interrupt.is_set():
                        break
                    raise
        finally:
            with self._lock:
                self._stream = None
            try:
                out.abort() if self._interrupt.is_set() else out.stop()
            except Exception as exc:  # noqa: BLE001
                log.debug("Error finalizing Kokoro stream: %s", exc)
            try:
                out.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("Error closing Kokoro stream: %s", exc)
