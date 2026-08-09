"""Ada assistant tools.

This package owns the module-level tool configuration and, via
:func:`load_all`, imports every tool submodule so their ``@tool``
decorators register with the registry.

Handlers must call :func:`get_config` lazily inside their bodies —
never at import time.
"""

from __future__ import annotations

import importlib
import logging

from ..config import Config
from . import registry
from .registry import ToolError

log = logging.getLogger("ada.tools")

_cfg: Config | None = None
_loaded = False

_SUBMODULES = (
    "apps",
    "browser",
    "calendar_mac",
    "clipboard",
    "context",
    "display",
    "files",
    "input_text",
    "memory_tools",
    "music",
    "productivity",
    "system",
    "web",
    "windows",
)


def configure(cfg: Config) -> None:
    """Set the configuration that tool handlers read via get_config()."""
    global _cfg
    _cfg = cfg


def get_config() -> Config:
    """Return the active Config. Raises if configure() has not run."""
    if _cfg is None:
        raise RuntimeError("Tools not configured")
    return _cfg


def load_all(cfg: Config) -> None:
    """Configure tools and import all tool submodules (idempotent)."""
    global _loaded
    configure(cfg)
    if _loaded:
        return
    for name in _SUBMODULES:
        importlib.import_module(f".{name}", __package__)
    _loaded = True
    log.debug("Loaded %d tools", len(registry.all_tools()))


__all__ = ["ToolError", "configure", "get_config", "load_all", "registry"]
