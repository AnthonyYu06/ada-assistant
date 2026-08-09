"""The brain: streaming tool loop, memory, prompts, TTS chunking.

Two interchangeable backends, selected by ``cfg.llm.provider``:
  - ``ollama``    — a local model via Ollama (free, on-device, no API key)
  - ``anthropic`` — the Claude Messages API (cloud, needs ANTHROPIC_API_KEY)

Both share the same safe tool-execution path (see ``base.BaseBrain``) and the
same ``handle(user_text, on_sentence, cancel) -> str`` interface, so the rest
of the assistant doesn't care which one is in use.
"""

from __future__ import annotations

from typing import Any

from ..config import Config
from ..safety.base import Confirmer, PermissionPolicy
from .base import BaseBrain
from .client import AnthropicBrain, Brain
from .memory import ConversationMemory
from .ollama_client import OllamaBrain
from .prompts import build_system_prompt
from .sentences import SentenceSplitter

__all__ = [
    "BaseBrain",
    "Brain",
    "AnthropicBrain",
    "OllamaBrain",
    "create_brain",
    "ConversationMemory",
    "SentenceSplitter",
    "build_system_prompt",
]


def create_brain(
    cfg: Config,
    policy: PermissionPolicy,
    confirmer: Confirmer,
    audit: Any,
) -> BaseBrain:
    """Build the brain for the configured provider (``cfg.llm.provider``)."""
    provider = cfg.llm.provider
    if provider == "ollama":
        return OllamaBrain(cfg, policy, confirmer, audit)
    if provider == "anthropic":
        return AnthropicBrain(cfg, policy, confirmer, audit)
    raise ValueError(
        f"Unknown llm.provider {provider!r} (expected 'ollama' or 'anthropic')"
    )
