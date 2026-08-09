"""JSONL audit log.

Every consequential event — tool calls, startup/shutdown, mute toggles,
wake activations, recognized command transcripts — is appended as one
JSON object per line. Only recognized COMMAND text is ever recorded for
transcripts; raw audio is never written anywhere.

The audit log must never take the assistant down: every failure is
logged and swallowed.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("ada.audit")

_MAX_STR_LEN = 300


def _truncate(value: Any) -> Any:
    """Truncate long strings (recursing into containers) to keep lines small."""
    if isinstance(value, str):
        if len(value) > _MAX_STR_LEN:
            return value[:_MAX_STR_LEN] + "…"
        return value
    if isinstance(value, dict):
        return {key: _truncate(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_truncate(item) for item in value]
    return value


class AuditLog:
    """Thread-safe, append-only JSONL event log. No-op when disabled."""

    def __init__(self, path: Path, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._lock = threading.Lock()
        if self.enabled:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                log.exception("Could not create audit log directory %s", self.path.parent)

    def record(self, kind: str, **fields: Any) -> None:
        """Append one event: {"ts": <UTC ISO 8601>, "kind": kind, **fields}."""
        if not self.enabled:
            return
        try:
            entry: dict[str, Any] = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "kind": kind,
            }
            for key, value in fields.items():
                entry[key] = _truncate(value)
            line = json.dumps(entry, ensure_ascii=False, default=str)
            with self._lock:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception:  # noqa: BLE001 - the audit log must never raise
            log.exception("Failed to write audit record (kind=%s)", kind)
