"""Configuration loading for Ada.

Configuration is layered:
  1. Bundled defaults (src/ada/config.default.yaml)
  2. A user config.yaml (explicit path, $ADA_CONFIG, ./config.yaml,
     or ~/.config/ada/config.yaml — first match wins)
  3. Secrets from the environment / a .env file next to the config
     (ANTHROPIC_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY).

Secrets are never read from YAML and never written to disk by this module.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("ada.config")

_SECRET_ENV_VARS = ("ANTHROPIC_API_KEY", "DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY")


@dataclass
class AssistantSection:
    name: str = "Ada"


@dataclass
class WakeSection:
    enabled: bool = False
    model: str = "hey_mycroft_v0.1"
    threshold: float = 0.5
    cooldown_s: float = 2.0


@dataclass
class AudioSection:
    input_device: int | str | None = None
    sample_rate: int = 16000
    frame_samples: int = 1280


@dataclass
class VadSection:
    threshold: float = 0.5
    pre_roll_ms: int = 320
    min_speech_ms: int = 200
    end_silence_ms: int = 600
    max_utterance_s: float = 15.0
    no_speech_timeout_s: float = 6.0


@dataclass
class LocalSTTSection:
    model: str = "small.en"
    compute_type: str = "int8"


@dataclass
class DeepgramSection:
    model: str = "nova-3"


@dataclass
class STTSection:
    provider: str = "local"
    local: LocalSTTSection = field(default_factory=LocalSTTSection)
    deepgram: DeepgramSection = field(default_factory=DeepgramSection)


@dataclass
class AnthropicLLMSection:
    model: str = "claude-opus-4-8"
    effort: str = "low"          # low | medium | high
    max_tokens: int = 2048


@dataclass
class OllamaLLMSection:
    # Any model tag you have pulled in Ollama (e.g. gemma4:12b, gemma4:e4b,
    # qwen2.5, llama3.1). Tool calling requires a model whose Ollama template
    # supports tools; if it doesn't, Ada falls back to chat-only.
    model: str = "gemma4:12b"
    host: str = "http://localhost:11434"
    temperature: float = 0.6
    num_ctx: int = 8192          # context window the model is loaded with
    keep_alive: str = "30m"      # keep the model resident in RAM between turns
    # Thinking models (gemma4, qwen3, deepseek-r1) silently "reason" for
    # seconds before answering — deadly for voice latency. False sends
    # "think": false (ignored gracefully by non-thinking models).
    think: bool = False


@dataclass
class LLMSection:
    provider: str = "ollama"     # ollama (local, free) | anthropic (cloud API)
    max_tool_iterations: int = 6
    history_max_messages: int = 24
    reset_history_after_min: int = 10
    anthropic: AnthropicLLMSection = field(default_factory=AnthropicLLMSection)
    ollama: OllamaLLMSection = field(default_factory=OllamaLLMSection)

    @property
    def active_model(self) -> str:
        """The model string for the currently selected provider."""
        if self.provider == "ollama":
            return self.ollama.model
        return self.anthropic.model


@dataclass
class SaySection:
    voice: str = "Samantha"
    rate_wpm: int = 190


@dataclass
class ElevenLabsSection:
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    model: str = "eleven_flash_v2_5"


@dataclass
class KokoroSection:
    # Local neural voice (needs the `voice` extra + a one-time model download).
    # Voices: af_heart / af_bella / af_nova / af_sarah (US female),
    # am_michael / am_adam / am_puck (US male), bf_emma / bm_george (British).
    voice: str = "af_heart"
    speed: float = 1.0
    lang: str = ""            # "" = auto (en-us, or en-gb for b* voices)


@dataclass
class TTSSection:
    provider: str = "say"    # say (macOS) | kokoro (local neural) | elevenlabs (cloud)
    say: SaySection = field(default_factory=SaySection)
    kokoro: KokoroSection = field(default_factory=KokoroSection)
    elevenlabs: ElevenLabsSection = field(default_factory=ElevenLabsSection)


@dataclass
class UISection:
    hud: bool = True           # native on-screen HUD window (needs the `gui` extra)
    theme: str = "jarvis-cyan"
    menubar: bool = True
    chimes: bool = True
    show_timings: bool = True  # per-turn latency readout in the HUD


@dataclass
class MemorySection:
    """Long-term memory: facts Ada remembers across sessions (JSONL on disk)."""

    enabled: bool = True
    path: str = "data/memory.jsonl"   # relative paths anchor to base_dir
    max_facts: int = 400
    inject_top_k: int = 4             # facts auto-recalled into each request


@dataclass
class HotkeySection:
    enabled: bool = True
    combo: str = "<cmd>+<shift>+a"


@dataclass
class PermissionsSection:
    allowed_app_dirs: list[str] = field(default_factory=lambda: [
        "/Applications", "/System/Applications", "/System/Library/CoreServices",
    ])
    blocked_apps: list[str] = field(default_factory=list)
    file_roots: list[str] = field(default_factory=lambda: [
        "~/Documents", "~/Desktop", "~/Downloads",
    ])
    url_schemes: list[str] = field(default_factory=lambda: ["http", "https"])
    always_confirm: list[str] = field(default_factory=lambda: ["quit_app", "system_sleep"])

    def file_roots_resolved(self) -> list[Path]:
        return [Path(p).expanduser().resolve() for p in self.file_roots]


@dataclass
class LoggingSection:
    dir: str = "logs"
    level: str = "INFO"
    audit: bool = True


@dataclass
class Config:
    assistant: AssistantSection = field(default_factory=AssistantSection)
    wake: WakeSection = field(default_factory=WakeSection)
    audio: AudioSection = field(default_factory=AudioSection)
    vad: VadSection = field(default_factory=VadSection)
    stt: STTSection = field(default_factory=STTSection)
    llm: LLMSection = field(default_factory=LLMSection)
    tts: TTSSection = field(default_factory=TTSSection)
    ui: UISection = field(default_factory=UISection)
    hotkey: HotkeySection = field(default_factory=HotkeySection)
    permissions: PermissionsSection = field(default_factory=PermissionsSection)
    memory: MemorySection = field(default_factory=MemorySection)
    logging: LoggingSection = field(default_factory=LoggingSection)

    # Where the user config lives; base_dir anchors relative paths (logs etc.).
    config_path: Path | None = None
    base_dir: Path = field(default_factory=Path.cwd)

    # Secrets — populated from the environment, never from YAML.
    anthropic_api_key: str | None = None
    deepgram_api_key: str | None = None
    elevenlabs_api_key: str | None = None

    @property
    def log_dir(self) -> Path:
        p = Path(self.logging.dir).expanduser()
        if not p.is_absolute():
            p = self.base_dir / p
        return p

    @property
    def memory_path(self) -> Path:
        p = Path(self.memory.path).expanduser()
        if not p.is_absolute():
            p = self.base_dir / p
        return p


def default_config_text() -> str:
    """The bundled default config file, as text (used by `ada setup`)."""
    return resources.files("ada").joinpath("config.default.yaml").read_text()


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _populate(obj: Any, data: dict, path: str = "") -> None:
    """Fill dataclass fields from a dict, warning about unknown keys."""
    fields = {f.name: f for f in obj.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    for key, value in data.items():
        if key not in fields:
            log.warning("Unknown config key: %s%s", path, key)
            continue
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _populate(current, value, path=f"{path}{key}.")
        else:
            setattr(obj, key, value)


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader. Never overrides variables already in the environment."""
    if not path.is_file():
        return
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError as exc:
        log.warning("Could not read %s: %s", path, exc)


def find_config_path(explicit: str | None = None) -> Path | None:
    """Locate the user config file. Returns None if only defaults apply."""
    candidates: list[Path] = []
    if explicit:
        return Path(explicit).expanduser()
    if os.environ.get("ADA_CONFIG"):
        candidates.append(Path(os.environ["ADA_CONFIG"]).expanduser())
    candidates.append(Path.cwd() / "config.yaml")
    candidates.append(Path.home() / ".config" / "ada" / "config.yaml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_config(path: str | None = None) -> Config:
    """Load configuration. `path` is an explicit config file (from --config)."""
    config_path = find_config_path(path)
    if path is not None and (config_path is None or not config_path.is_file()):
        raise FileNotFoundError(f"Config file not found: {path}")

    merged = yaml.safe_load(default_config_text()) or {}
    if config_path is not None:
        user_data = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(user_data, dict):
            raise ValueError(f"{config_path} does not contain a YAML mapping")
        merged = _deep_merge(merged, user_data)

    cfg = Config()
    _populate(cfg, merged)
    cfg.config_path = config_path
    cfg.base_dir = config_path.parent if config_path else Path.cwd()

    # Secrets: .env next to the config (and cwd), then the environment.
    _load_dotenv(cfg.base_dir / ".env")
    if cfg.base_dir != Path.cwd():
        _load_dotenv(Path.cwd() / ".env")
    cfg.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY") or None
    cfg.deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY") or None
    cfg.elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY") or None
    return cfg
