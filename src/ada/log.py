"""Logging setup for Ada.

- App log: human-readable, to stderr and logs/ada.log
- Audit log: see ada.safety.audit (JSONL of every tool call)
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from .config import Config


def setup_logging(cfg: Config, console: bool = True) -> Path:
    """Configure the root 'ada' logger. Returns the log directory."""
    log_dir = cfg.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("ada")
    root.setLevel(getattr(logging, cfg.logging.level.upper(), logging.INFO))
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "ada.log", maxBytes=2_000_000, backupCount=3
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        stream_handler.setLevel(logging.WARNING)
        root.addHandler(stream_handler)

    return log_dir
