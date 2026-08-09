"""A tiny thread-safe pub/sub bus for UI surfaces.

The assistant runs on worker threads; UI surfaces (the HUD, the console) live
elsewhere. Rather than couple the pipeline to any one UI, the pipeline
*publishes* small events here and any UI *subscribes*.

Event kinds and payloads:
  - "user_text"        -> str    the recognized/typed user utterance
  - "assistant_delta"  -> str    one streamed sentence of the reply
  - "assistant_done"   -> str    the full reply text (or "" on error/clear)
  - "level"            -> float  0..1 audio level (optional, for the orb)
  - "tool"             -> dict   tool activity for the HUD's chips:
                                 {name, summary, state: "running"|"ok"|"error"}
  - "timing"           -> dict   per-turn latency breakdown:
                                 {stages: [[label, ms], ...], total_ms}
  - "display"          -> dict   a display card the HUD renders graphically
                                 (schema below)

Display cards
-------------
A display card is a plain JSON-able dict: {"kind": ..., "title"?: str,
"id"?: str, ...kind-specific fields}. Cards with the same "id" REPLACE each
other in the HUD (live updates — e.g. a ticking timer); cards without an id
are appended. Kinds and their fields:

  markdown     {body: str}          mini-markdown: #/## headings, **bold**,
                                    *italic*, `code`, - bullets, 1. lists,
                                    [text](url)
  table        {columns: [str], rows: [[str, ...], ...]}
  list         {items: [{text: str, sub?: str, icon?: str}]}   icon = emoji
  keyvalue     {pairs: [[key, value], ...]}
  chart        {chart: "bar"|"line", labels: [str], values: [num], unit?: str}
  image        {src: <data: URI>, caption?: str}
  web_results  {results: [{title, url, snippet?}]}
  files        {items: [{name, path, kind?}]}
  now_playing  {track, artist, album?, state: "playing"|"paused", app?}
  timer        {label: str, ends_at: <unix seconds>, duration_s: num}
  calendar     {events: [{title, start, end?, location?, calendar?}]}

The HUD must tolerate unknown kinds (fallback: show the title and a plain
"body" field if present) so new kinds can be added without breaking old UIs.

Status changes are NOT sent here — they flow through AssistantState's own
listener mechanism. Subscriber callbacks must never raise; if one does, it is
logged and the others still run.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

log = logging.getLogger("ada.ui.events")

Listener = Callable[[str, Any], None]

_lock = threading.Lock()
_listeners: list[Listener] = []


def subscribe(listener: Listener) -> None:
    with _lock:
        if listener not in _listeners:
            _listeners.append(listener)


def unsubscribe(listener: Listener) -> None:
    with _lock:
        if listener in _listeners:
            _listeners.remove(listener)


def publish(kind: str, payload: Any = None) -> None:
    with _lock:
        listeners = list(_listeners)
    for listener in listeners:
        try:
            listener(kind, payload)
        except Exception:  # noqa: BLE001 - a broken UI must not kill the pipeline
            log.exception("UI event listener failed for %r", kind)


def reset() -> None:
    """Testing helper — drop all subscribers."""
    with _lock:
        _listeners.clear()
