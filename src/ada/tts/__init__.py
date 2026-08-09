"""Text-to-speech: provider contract, providers, and the sentence-queue Speaker."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import TTSProvider
from .say import SayTTS
from .speaker import Speaker

if TYPE_CHECKING:
    from ..config import Config

log = logging.getLogger("ada.tts")

__all__ = ["TTSProvider", "Speaker", "SayTTS", "create_tts"]


def create_tts(cfg: Config) -> TTSProvider:
    """Build the configured TTS provider.

    'kokoro' (local neural voice) and 'elevenlabs' (cloud) both fall back to
    the built-in macOS 'say' voice — with a clear warning — when they aren't
    ready (missing package, un-downloaded model, or no API key), so the
    assistant always has a voice.
    """
    provider = str(cfg.tts.provider).strip().lower()

    def _say() -> TTSProvider:
        return SayTTS(cfg.tts.say.voice, cfg.tts.say.rate_wpm)

    if provider == "kokoro":
        try:
            import kokoro_onnx  # noqa: F401
        except Exception:  # noqa: BLE001
            log.warning(
                "tts.provider is 'kokoro' but kokoro-onnx isn't installed "
                "(pip install -e \".[voice]\"); falling back to macOS 'say'"
            )
            return _say()
        from .kokoro import KokoroTTS, models_present

        if not models_present():
            log.warning(
                "tts.provider is 'kokoro' but the voice model isn't downloaded "
                "(run `ada setup`); falling back to macOS 'say'"
            )
            return _say()
        return KokoroTTS(
            cfg.tts.kokoro.voice,
            cfg.tts.kokoro.speed,
            cfg.tts.kokoro.lang or None,
        )

    if provider == "elevenlabs":
        if not cfg.elevenlabs_api_key:
            log.warning(
                "tts.provider is 'elevenlabs' but ELEVENLABS_API_KEY is not set; "
                "falling back to macOS 'say'"
            )
            return SayTTS(cfg.tts.say.voice, cfg.tts.say.rate_wpm)
        from .elevenlabs import ElevenLabsTTS

        return ElevenLabsTTS(
            cfg.elevenlabs_api_key,
            cfg.tts.elevenlabs.voice_id,
            cfg.tts.elevenlabs.model,
        )

    if provider == "say":
        return _say()

    raise ValueError(
        f"Unknown TTS provider: {cfg.tts.provider!r} "
        "(expected 'say', 'kokoro', or 'elevenlabs')"
    )
