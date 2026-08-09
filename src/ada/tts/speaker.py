"""Sentence-queue playback manager.

The brain streams its reply sentence by sentence; each sentence is enqueued
here and a background worker speaks them in order through the configured
TTS provider. stop() flushes everything (barge-in / kill switch) while
leaving the Speaker reusable for the next reply.

Optional pipelining capability: a provider may expose, in addition to the
TTSProvider protocol, ``synthesize(text) -> Any`` (expensive preparation —
e.g. model inference — returning an opaque prepared-audio object) and
``play_prepared(prepared) -> None`` (playback of that object, honoring the
provider's interrupt flag exactly like speak()). When the provider hasattr
BOTH, playback is pipelined: while sentence N plays, sentence N+1 is
synthesized on a single background prefetch thread (exactly one synthesis at
a time, exactly one chunk ahead), hiding the ~0.5x-realtime synthesis gap
between sentences. Providers without the capability keep the exact serial
speak() path. Barge-in semantics are identical in both modes: the generation
tag decides at PLAY time, under the Speaker lock, atomically with arm() — a
prefetched-but-unplayed synthesis for a stopped generation is discarded,
never played.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable

from .base import TTSProvider

log = logging.getLogger("ada.tts.speaker")

# Queue items are (generation, text); None is the shutdown sentinel.
_Item = tuple[int, str] | None

# Prefetch handoff items are (generation, text, prepared); None is the
# forwarded shutdown sentinel.
_Prepared = tuple[int, str, Any] | None

# Marker (identity-compared) for a chunk whose synthesis failed or was
# skipped as stale: nothing to play, but the chunk still flows through the
# worker so its _busy count is released exactly once.
_NO_AUDIO = object()


class Speaker:
    """Serializes TTS playback of streamed text chunks on a worker thread."""

    def __init__(
        self,
        provider: TTSProvider,
        on_speaking_change: Callable[[bool], None] | None = None,
    ) -> None:
        self._provider = provider
        self._on_speaking_change = on_speaking_change
        self._queue: queue.Queue[_Item] = queue.Queue()
        self._cond = threading.Condition()
        # Chunks enqueued but not yet fully played (includes the one the
        # worker is currently speaking and, in pipelined mode, any chunk the
        # prefetch thread has popped for synthesis).
        self._busy = 0
        # Bumped by stop()/close(); chunks from an older generation are
        # discarded instead of spoken.
        self._gen = 0
        self._speaking = False  # last value reported via on_speaking_change
        self._closed = False
        # Pipelining capability (see module docstring): when the provider
        # exposes BOTH synthesize() and play_prepared(), a single prefetch
        # thread becomes the sole consumer of self._queue. It synthesizes
        # chunks in order and hands (gen, text, prepared) tuples — and the
        # forwarded shutdown sentinel — to the worker via self._prepared_q,
        # throttled to exactly one chunk ahead of playback (task_done/join).
        # Every popped chunk is forwarded, even on synthesis failure or when
        # already stale, so the worker releases each chunk's _busy count
        # exactly once and playback order is preserved.
        pipelined = callable(getattr(provider, "synthesize", None)) and callable(
            getattr(provider, "play_prepared", None)
        )
        self._prepared_q: queue.Queue[_Prepared] = queue.Queue()
        self._prefetch: threading.Thread | None = None
        if pipelined:
            self._prefetch = threading.Thread(
                target=self._prefetch_run, name="ada-speaker-prefetch", daemon=True
            )
            self._prefetch.start()
        self._worker = threading.Thread(
            target=self._run, name="ada-speaker", daemon=True
        )
        self._worker.start()

    # -- public API -------------------------------------------------------
    def current_generation(self) -> int:
        """Return the live generation (bumped by stop()/close())."""
        with self._cond:
            return self._gen

    def enqueue(self, text: str, generation: int | None = None) -> None:
        """Queue one chunk of text for playback. Blank chunks are ignored.

        If `generation` is given and no longer matches the live generation
        (i.e. a stop()/close() has happened since the caller read it), the
        chunk is dropped instead of queued. This lets a caller tie a chunk to
        the turn that produced it, so a sentence emitted by a barged-in turn
        cannot play during the next turn. `generation=None` uses the live
        generation (unconditional enqueue).
        """
        if not text or not text.strip():
            return
        with self._cond:
            if self._closed:
                log.warning("enqueue() after close(); dropping chunk")
                return
            if generation is not None and generation != self._gen:
                return  # chunk belongs to a superseded generation; drop it
            self._busy += 1
            self._queue.put((self._gen, text))

    def wait_until_done(self, timeout: float | None = None) -> bool:
        """Block until the queue is drained and playback has ended.

        Returns True if everything finished, False on timeout.
        """
        with self._cond:
            return self._cond.wait_for(lambda: self._busy == 0, timeout=timeout)

    def stop(self) -> None:
        """Flush all queued chunks and interrupt the current one.

        The worker stays alive; the Speaker is immediately reusable.
        """
        with self._cond:
            if self._closed:
                return  # already shut down; don't double-bump or eat the sentinel
            self._gen += 1
            drained = self._drain_queue_locked()
            self._busy -= drained
            # provider.stop() must run under the lock so it cannot interleave
            # between the worker's arm() and its provider.speak() /
            # play_prepared() call. A chunk already popped by the prefetch
            # thread is NOT in the queue anymore: it is discarded by the
            # worker's play-time generation check instead of drained here.
            self._provider.stop()
            self._cond.notify_all()

    @property
    def is_speaking(self) -> bool:
        """True while any chunk is queued or currently being played."""
        with self._cond:
            return self._busy > 0

    def close(self, timeout: float | None = 5.0) -> None:
        """Shut the worker down cleanly (used on exit). Idempotent."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._gen += 1
            drained = self._drain_queue_locked()
            self._busy -= drained
            # In pipelined mode the sentinel is consumed by the prefetch
            # thread, which forwards it to the worker and exits; in serial
            # mode the worker consumes it directly. Either way both threads
            # see exactly one shutdown signal.
            self._queue.put(None)  # wake the worker so it can exit
            self._cond.notify_all()
        self._provider.stop()
        self._worker.join(timeout)
        if self._worker.is_alive():
            log.warning("Speaker worker did not exit within %.1fs", timeout or 0.0)
        if self._prefetch is not None:
            # Any in-flight synthesis finishes (model inference is not
            # interruptible), its result is forwarded (and discarded by the
            # worker's generation check), then the sentinel is forwarded and
            # the thread exits.
            self._prefetch.join(timeout)
            if self._prefetch.is_alive():
                log.warning(
                    "Speaker prefetch thread did not exit within %.1fs",
                    timeout or 0.0,
                )

    # -- internals --------------------------------------------------------
    def _drain_queue_locked(self) -> int:
        """Discard queued chunks; return the count of real (non-sentinel) chunks.

        The close() shutdown sentinel (None) is never consumed here: if we pop
        it, we re-queue it and do not count it, so a stop() racing with close()
        cannot swallow the sentinel (which would leave the worker unable to
        exit) nor drive _busy negative.
        """
        drained = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return drained
            if item is None:
                self._queue.put(None)  # keep the shutdown sentinel intact
                return drained
            drained += 1

    def _run(self) -> None:
        if self._prefetch is not None:
            self._run_pipelined()
        else:
            self._run_serial()

    def _run_serial(self) -> None:
        """Worker loop for plain speak()-only providers (unchanged path)."""
        while True:
            item = self._queue.get()
            if item is None:  # close() sentinel
                self._set_speaking(False)
                return
            gen, text = item
            # Decide liveness and arm the provider atomically under the same
            # lock that stop() uses to set the interrupt, so a stop() landing
            # here cannot be lost between arm() and speak() (TOCTOU fix).
            with self._cond:
                live = gen == self._gen
                if live:
                    self._provider.arm()
            if live:
                self._set_speaking(True)
                try:
                    self._provider.speak(text)
                except Exception:  # noqa: BLE001 - provider failures must not kill the worker
                    log.exception(
                        "TTS provider %r failed while speaking",
                        getattr(self._provider, "name", "?"),
                    )
            with self._cond:
                self._busy -= 1
                idle = self._busy <= 0
                self._cond.notify_all()
            if idle:
                self._set_speaking(False)

    def _run_pipelined(self) -> None:
        """Worker loop for synthesize()/play_prepared() providers.

        Consumes ready-to-play chunks from the prefetch thread instead of raw
        text from self._queue — that is the ONLY difference from the serial
        path. The barge-in contract is unchanged: the generation tag decides
        at PLAY time, under the Speaker lock, atomically with arm() — a
        prefetched synthesis whose generation went stale (stop()/close()
        landed between synthesis and playback) is discarded, never played,
        and a stop() during play_prepared() interrupts it via provider.stop()
        exactly as it interrupts speak().
        """
        while True:
            item = self._prepared_q.get()
            # Unblocks the prefetch thread's one-ahead throttle: it may now
            # synthesize the NEXT chunk while this one plays.
            self._prepared_q.task_done()
            if item is None:  # close() sentinel, forwarded by the prefetch thread
                self._set_speaking(False)
                return
            gen, text, prepared = item
            # Decide liveness and arm the provider atomically under the same
            # lock that stop() uses to set the interrupt, so a stop() landing
            # here cannot be lost between arm() and play_prepared() (same
            # TOCTOU protocol as the serial path).
            with self._cond:
                live = gen == self._gen and prepared is not _NO_AUDIO
                if live:
                    self._provider.arm()
            if live:
                self._set_speaking(True)
                try:
                    self._provider.play_prepared(prepared)
                except Exception:  # noqa: BLE001 - provider failures must not kill the worker
                    log.exception(
                        "TTS provider %r failed while playing prepared audio",
                        getattr(self._provider, "name", "?"),
                    )
            with self._cond:
                self._busy -= 1
                idle = self._busy <= 0
                self._cond.notify_all()
            if idle:
                self._set_speaking(False)

    def _prefetch_run(self) -> None:
        """Prefetch thread: synthesize upcoming chunks during playback.

        Sole consumer of self._queue in pipelined mode (stop()/close() still
        drain it concurrently — queue.Queue is thread-safe, and a chunk they
        cannot drain because it was already popped here is discarded by the
        worker's play-time generation check instead). Every popped chunk is
        forwarded to the worker — even stale or failed ones, as _NO_AUDIO —
        preserving order and releasing each chunk's _busy count exactly once.
        The task_done()/join() pair throttles lookahead: synthesis of chunk
        N+1 starts once the worker picks up chunk N for playback, so there is
        never more than one synthesis running and never more than one
        prepared chunk waiting.
        """
        while True:
            item = self._queue.get()
            if item is None:  # close() sentinel: hand it on to the worker
                self._prepared_q.put(None)
                return
            gen, text = item
            with self._cond:
                stale = gen != self._gen
            if stale:
                # A stop()/close() superseded this chunk before synthesis
                # began: skip the expensive inference. Only an optimization —
                # the authoritative liveness check happens at play time.
                prepared: Any = _NO_AUDIO
            else:
                try:
                    prepared = self._provider.synthesize(text)
                except Exception:  # noqa: BLE001 - synth failures must not kill the pipeline
                    log.exception(
                        "TTS provider %r failed to synthesize; skipping chunk",
                        getattr(self._provider, "name", "?"),
                    )
                    prepared = _NO_AUDIO
            self._prepared_q.put((gen, text, prepared))
            self._prepared_q.join()  # one-ahead throttle (see docstring)

    def _set_speaking(self, value: bool) -> None:
        with self._cond:
            if self._speaking == value:
                return
            self._speaking = value
        if self._on_speaking_change is not None:
            try:
                self._on_speaking_change(value)
            except Exception:  # noqa: BLE001 - UI callbacks must not kill the worker
                log.exception("on_speaking_change callback failed")
