"""Speech-to-text providers.

Use create_stt(cfg) to build the provider selected in the config
(local faster-whisper by default, Deepgram if configured and keyed).
"""

from __future__ import annotations

import logging

from ..config import Config
from .base import STTProvider, STTResult

__all__ = ["STTProvider", "STTResult", "create_stt"]

log = logging.getLogger("ada.stt")


def create_stt(cfg: Config) -> STTProvider:
    """Build the STT provider selected by cfg.stt.provider."""
    provider = cfg.stt.provider
    if provider == "deepgram":
        if cfg.deepgram_api_key:
            from .deepgram import DeepgramSTT

            return DeepgramSTT(api_key=cfg.deepgram_api_key, model=cfg.stt.deepgram.model)
        log.warning("DEEPGRAM_API_KEY not set, falling back to local Whisper")
        provider = "local"
    if provider == "local":
        from .local_whisper import LocalWhisperSTT

        return LocalWhisperSTT(
            model_name=cfg.stt.local.model,
            compute_type=cfg.stt.local.compute_type,
        )
    raise ValueError(f"Unknown STT provider: {cfg.stt.provider!r}")
