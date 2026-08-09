"""Lightweight per-turn latency instrumentation.

A TurnTimer is created at the start of a conversational turn; the pipeline
calls mark() as each stage completes (record, transcribe, first-sentence,
speak-start, ...). finish() logs the breakdown and publishes a "timing"
event on the UI bus so the HUD can show how fast the turn was.

Marks measure the time since the PREVIOUS mark, so the stages list reads as
a waterfall. All methods are cheap and never raise.
"""

from __future__ import annotations

import logging
import time

from .ui import events as ui_events

log = logging.getLogger("ada.perf")


class TurnTimer:
    """Stage-by-stage wall-clock breakdown of one conversational turn."""

    def __init__(self, publish: bool = True) -> None:
        self._publish = publish
        self._t0 = time.monotonic()
        self._last = self._t0
        self._stages: list[tuple[str, float]] = []
        self._finished = False

    def mark(self, stage: str) -> float:
        """Record `stage` as completed now; returns its duration in ms."""
        now = time.monotonic()
        ms = (now - self._last) * 1000.0
        self._last = now
        self._stages.append((stage, round(ms, 1)))
        return ms

    def stages(self) -> list[tuple[str, float]]:
        return list(self._stages)

    def total_ms(self) -> float:
        return round((self._last - self._t0) * 1000.0, 1)

    def finish(self) -> None:
        """Log the waterfall and publish it to the UI (once)."""
        if self._finished or not self._stages:
            return
        self._finished = True
        total = self.total_ms()
        try:
            log.info(
                "turn timing: %s = %.0fms total",
                " + ".join(f"{name} {ms:.0f}ms" for name, ms in self._stages),
                total,
            )
            if self._publish:
                ui_events.publish(
                    "timing",
                    {
                        "stages": [[name, ms] for name, ms in self._stages],
                        "total_ms": total,
                    },
                )
        except Exception:  # noqa: BLE001 - instrumentation must never break a turn
            log.exception("Failed to publish turn timing")
