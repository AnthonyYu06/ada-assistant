"""Tests for ada.tts.speaker.Speaker: ordering, waiting, barge-in, shutdown."""

from __future__ import annotations

import threading
import time
from typing import Iterator

import pytest

from ada.tts.speaker import Speaker

_TIMEOUT = 5.0  # generous ceiling so slow CI never flakes


def _wait_for(predicate, timeout: float = _TIMEOUT, interval: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


class FakeTTS:
    """TTSProvider double whose playback duration is controlled by events."""

    name = "fake"

    def __init__(self) -> None:
        self.started: list[str] = []  # chunks whose playback began
        self.spoken: list[str] = []   # chunks whose playback finished
        self.stop_calls = 0
        self._lock = threading.Lock()
        self._release = threading.Event()
        self._interrupt = threading.Event()
        self._blocking = False

    def hold(self) -> None:
        """Make speak() block until release() or stop() is called."""
        self._blocking = True
        self._release.clear()
        self._interrupt.clear()

    def release(self) -> None:
        """Let any blocked (and all future) speak() calls finish."""
        self._blocking = False
        self._release.set()

    # -- TTSProvider protocol ----------------------------------------------
    def arm(self) -> None:
        """Reset the per-utterance interrupt state (Speaker calls this before
        committing to a live chunk). speak() no longer clears it itself."""
        with self._lock:
            self._interrupt.clear()

    def speak(self, text: str) -> None:
        with self._lock:
            self.started.append(text)
        if self._blocking:
            while not self._release.is_set() and not self._interrupt.is_set():
                time.sleep(0.002)
        with self._lock:
            self.spoken.append(text)

    def stop(self) -> None:
        with self._lock:
            self.stop_calls += 1
        self._interrupt.set()


class FakePipelinedTTS(FakeTTS):
    """Provider double exposing the optional synthesize/play_prepared
    capability. Records per-call order and monotonic timestamps so tests can
    prove that synthesis of chunk N+1 overlaps playback of chunk N, and that
    prepared audio made stale by stop() is discarded, never played."""

    name = "fake-pipelined"

    def __init__(self) -> None:
        super().__init__()
        self.timeline: list[tuple[float, str, str]] = []  # (t, op, text)
        self.playing: str | None = None  # chunk currently inside play_prepared
        self.synth_fail: set[str] = set()  # texts whose synthesize() raises
        self._hold_synth = threading.Event()
        self._hold_synth.set()  # released by default

    # -- test controls -------------------------------------------------------
    def hold_synth(self) -> None:
        """Make synthesize() block until release_synth() is called."""
        self._hold_synth.clear()

    def release_synth(self) -> None:
        self._hold_synth.set()

    def ops(self, op: str) -> list[str]:
        with self._lock:
            return [text for _, o, text in self.timeline if o == op]

    def op_time(self, op: str, text: str) -> float:
        with self._lock:
            return next(t for t, o, x in self.timeline if o == op and x == text)

    def _record(self, op: str, text: str) -> None:
        with self._lock:
            self.timeline.append((time.monotonic(), op, text))

    # -- optional pipelining capability ---------------------------------------
    def synthesize(self, text: str) -> tuple[str, str]:
        self._record("synth_start", text)
        self._hold_synth.wait(_TIMEOUT)
        if text in self.synth_fail:
            self._record("synth_fail", text)
            raise RuntimeError("synthetic synthesis failure")
        self._record("synth_done", text)
        return ("prepared", text)

    def play_prepared(self, prepared: tuple[str, str]) -> None:
        tag, text = prepared
        assert tag == "prepared", "play_prepared must receive synthesize()'s object"
        with self._lock:
            self.playing = text
            self.started.append(text)
        self._record("play_start", text)
        if self._blocking:
            while not self._release.is_set() and not self._interrupt.is_set():
                time.sleep(0.002)
        self._record("play_done", text)
        with self._lock:
            self.playing = None
            self.spoken.append(text)

    def speak(self, text: str) -> None:  # pragma: no cover - must never run
        raise AssertionError(
            "Speaker must use the synthesize/play_prepared pipeline, not speak()"
        )


@pytest.fixture
def fake() -> FakeTTS:
    return FakeTTS()


@pytest.fixture
def pipelined() -> Iterator[tuple[FakePipelinedTTS, Speaker]]:
    fake = FakePipelinedTTS()
    speaker = Speaker(fake)
    yield fake, speaker
    fake.release()
    fake.release_synth()
    speaker.close(timeout=_TIMEOUT)


@pytest.fixture
def speaker(fake: FakeTTS) -> Iterator[Speaker]:
    spk = Speaker(fake)
    yield spk
    fake.release()
    spk.close(timeout=_TIMEOUT)


def test_enqueue_order_preserved(fake: FakeTTS, speaker: Speaker) -> None:
    texts = [f"Sentence number {i}." for i in range(5)]
    for text in texts:
        speaker.enqueue(text)
    assert speaker.wait_until_done(timeout=_TIMEOUT)
    assert fake.spoken == texts


def test_blank_chunks_are_ignored(fake: FakeTTS, speaker: Speaker) -> None:
    speaker.enqueue("")
    speaker.enqueue("   \n ")
    assert speaker.wait_until_done(timeout=_TIMEOUT)
    assert fake.started == []
    assert speaker.is_speaking is False


def test_wait_until_done_blocks_until_played(fake: FakeTTS, speaker: Speaker) -> None:
    fake.hold()
    speaker.enqueue("Hello there.")
    assert _wait_for(lambda: fake.started == ["Hello there."])
    # Playback is still in progress: a short wait must time out...
    assert speaker.wait_until_done(timeout=0.2) is False
    assert speaker.is_speaking is True
    # ...and once the provider finishes, the wait succeeds.
    fake.release()
    assert speaker.wait_until_done(timeout=_TIMEOUT) is True
    assert fake.spoken == ["Hello there."]
    assert speaker.is_speaking is False


def test_stop_flushes_queue_mid_playback(fake: FakeTTS, speaker: Speaker) -> None:
    fake.hold()
    speaker.enqueue("First chunk playing.")
    speaker.enqueue("Second chunk queued.")
    speaker.enqueue("Third chunk queued.")
    assert _wait_for(lambda: fake.started == ["First chunk playing."])

    speaker.stop()

    assert fake.stop_calls >= 1  # the in-flight chunk was interrupted
    assert speaker.wait_until_done(timeout=_TIMEOUT) is True
    assert _wait_for(lambda: not speaker.is_speaking)
    # The queued chunks were discarded, never spoken.
    assert fake.started == ["First chunk playing."]

    # The Speaker is immediately reusable after stop().
    fake.release()
    speaker.enqueue("Back again.")
    assert speaker.wait_until_done(timeout=_TIMEOUT)
    assert "Back again." in fake.spoken


def test_is_speaking_transitions(fake: FakeTTS) -> None:
    events: list[bool] = []
    speaker = Speaker(fake, on_speaking_change=events.append)
    try:
        assert speaker.is_speaking is False
        fake.hold()
        speaker.enqueue("Talking now.")
        assert _wait_for(lambda: speaker.is_speaking)
        assert _wait_for(lambda: events == [True])
        fake.release()
        assert speaker.wait_until_done(timeout=_TIMEOUT)
        assert _wait_for(lambda: events == [True, False])
        assert speaker.is_speaking is False
    finally:
        fake.release()
        speaker.close(timeout=_TIMEOUT)


def test_provider_failure_does_not_kill_worker(speaker: Speaker, fake: FakeTTS) -> None:
    original_speak = fake.speak

    def flaky_speak(text: str) -> None:
        if text == "boom":
            raise RuntimeError("synthetic provider failure")
        original_speak(text)

    fake.speak = flaky_speak  # type: ignore[method-assign]
    speaker.enqueue("boom")
    speaker.enqueue("Still alive.")
    assert speaker.wait_until_done(timeout=_TIMEOUT)
    assert fake.spoken == ["Still alive."]


def test_close_terminates_worker(fake: FakeTTS) -> None:
    speaker = Speaker(fake)
    speaker.enqueue("Quick one.")
    assert speaker.wait_until_done(timeout=_TIMEOUT)
    speaker.close(timeout=_TIMEOUT)
    assert speaker._worker.is_alive() is False
    # close() is idempotent.
    speaker.close(timeout=_TIMEOUT)
    # enqueue() after close is dropped, not an error.
    speaker.enqueue("Too late.")
    assert "Too late." not in fake.started
    assert speaker.is_speaking is False


def test_close_interrupts_in_flight_playback(fake: FakeTTS) -> None:
    speaker = Speaker(fake)
    fake.hold()
    speaker.enqueue("Long monologue.")
    assert _wait_for(lambda: fake.started == ["Long monologue."])
    speaker.close(timeout=_TIMEOUT)  # stop() unblocks the provider
    assert speaker._worker.is_alive() is False
    assert fake.stop_calls >= 1


def test_stale_generation_chunk_is_dropped(fake: FakeTTS, speaker: Speaker) -> None:
    # A caller reads the generation, then a barge-in (stop) bumps it.
    gen = speaker.current_generation()
    speaker.stop()
    assert speaker.current_generation() != gen

    # A chunk tied to the now-superseded generation must never be spoken.
    speaker.enqueue("Stale sentence from the barged-in turn.", generation=gen)
    assert speaker.wait_until_done(timeout=_TIMEOUT) is True
    assert "Stale sentence from the barged-in turn." not in fake.started
    assert speaker.is_speaking is False

    # A chunk tied to the live generation still plays normally.
    live = speaker.current_generation()
    speaker.enqueue("Fresh sentence for the new turn.", generation=live)
    assert speaker.wait_until_done(timeout=_TIMEOUT) is True
    assert "Fresh sentence for the new turn." in fake.spoken


def test_stop_after_close_is_noop(fake: FakeTTS) -> None:
    speaker = Speaker(fake)
    speaker.enqueue("Only sentence.")
    assert speaker.wait_until_done(timeout=_TIMEOUT)
    speaker.close(timeout=_TIMEOUT)
    assert speaker._worker.is_alive() is False

    # stop() after close() must not eat the shutdown sentinel or drive the
    # busy count negative: the worker stays exited and waits succeed.
    speaker.stop()
    assert speaker._worker.is_alive() is False
    assert speaker.wait_until_done(timeout=_TIMEOUT) is True


# -- pipelined providers (optional synthesize/play_prepared capability) --------


def test_pipelined_synthesis_overlaps_playback(
    pipelined: tuple[FakePipelinedTTS, Speaker],
) -> None:
    fake, speaker = pipelined
    fake.hold()  # play_prepared() blocks until release()/stop()
    speaker.enqueue("First sentence.")
    assert _wait_for(lambda: fake.playing == "First sentence.")
    # Second sentence arrives while the first is already playing — the
    # common streaming case — and must synthesize DURING that playback.
    speaker.enqueue("Second sentence.")
    assert _wait_for(lambda: "Second sentence." in fake.ops("synth_done"))
    assert fake.playing == "First sentence."  # still mid-playback: true overlap
    fake.release()
    assert speaker.wait_until_done(timeout=_TIMEOUT)
    assert fake.spoken == ["First sentence.", "Second sentence."]
    # The timestamps agree: synthesis of 2 started before playback of 1 ended.
    assert fake.op_time("synth_start", "Second sentence.") < fake.op_time(
        "play_done", "First sentence."
    )
    # The pipeline was used; the serial speak() path never ran (the fake's
    # speak() raises if called).


def test_stop_between_synth_and_play_discards_prepared_audio(
    pipelined: tuple[FakePipelinedTTS, Speaker],
) -> None:
    fake, speaker = pipelined
    fake.hold()
    speaker.enqueue("Now playing.")
    assert _wait_for(lambda: fake.playing == "Now playing.")
    speaker.enqueue("Prefetched, never played.")
    assert _wait_for(lambda: "Prefetched, never played." in fake.ops("synth_done"))

    # Barge-in lands after chunk 2 was synthesized but before it played.
    speaker.stop()

    assert speaker.wait_until_done(timeout=_TIMEOUT) is True
    assert "Prefetched, never played." not in fake.ops("play_start")

    # The Speaker is immediately reusable for the next turn.
    fake.release()
    speaker.enqueue("Next turn.")
    assert speaker.wait_until_done(timeout=_TIMEOUT)
    assert fake.ops("play_start")[-1] == "Next turn."


def test_stop_skips_synthesis_of_queued_chunks(
    pipelined: tuple[FakePipelinedTTS, Speaker],
) -> None:
    # stop() while chunk 1 is mid-synthesis: chunks already drained from the
    # queue are never synthesized, and the in-flight one is never played.
    fake, speaker = pipelined
    fake.hold_synth()
    speaker.enqueue("Mid synthesis.")
    speaker.enqueue("Still queued.")
    assert _wait_for(lambda: "Mid synthesis." in fake.ops("synth_start"))

    speaker.stop()
    fake.release_synth()

    assert speaker.wait_until_done(timeout=_TIMEOUT) is True
    assert fake.ops("play_start") == []
    assert "Still queued." not in fake.ops("synth_start")


def test_pipelined_stale_generation_chunk_is_dropped(
    pipelined: tuple[FakePipelinedTTS, Speaker],
) -> None:
    fake, speaker = pipelined
    gen = speaker.current_generation()
    speaker.stop()
    # A chunk tied to the superseded generation is dropped, exactly as in the
    # serial path — it is neither synthesized nor played.
    speaker.enqueue("Stale sentence.", generation=gen)
    assert speaker.wait_until_done(timeout=_TIMEOUT) is True
    assert fake.ops("synth_start") == []
    assert fake.ops("play_start") == []

    speaker.enqueue("Fresh sentence.", generation=speaker.current_generation())
    assert speaker.wait_until_done(timeout=_TIMEOUT) is True
    assert fake.ops("play_done") == ["Fresh sentence."]


def test_pipelined_synthesis_error_skips_chunk_and_continues(
    pipelined: tuple[FakePipelinedTTS, Speaker],
) -> None:
    fake, speaker = pipelined
    fake.synth_fail.add("boom")
    # Failure on the stream's first chunk AND on a prefetched later chunk:
    # both are logged + skipped; the pipeline keeps playing everything else.
    speaker.enqueue("boom")
    speaker.enqueue("First survivor.")
    speaker.enqueue("boom")
    speaker.enqueue("Second survivor.")
    assert speaker.wait_until_done(timeout=_TIMEOUT)
    assert fake.ops("play_done") == ["First survivor.", "Second survivor."]
    assert fake.ops("play_start") == ["First survivor.", "Second survivor."]


def test_pipelined_close_joins_both_threads(
    pipelined: tuple[FakePipelinedTTS, Speaker],
) -> None:
    fake, speaker = pipelined
    speaker.enqueue("One.")
    speaker.enqueue("Two.")
    assert speaker.wait_until_done(timeout=_TIMEOUT)
    speaker.close(timeout=_TIMEOUT)
    assert speaker._worker.is_alive() is False
    assert speaker._prefetch is not None
    assert speaker._prefetch.is_alive() is False
    # close() stays idempotent with the prefetch thread in place.
    speaker.close(timeout=_TIMEOUT)


def test_pipelined_close_interrupts_playback_and_joins(
    pipelined: tuple[FakePipelinedTTS, Speaker],
) -> None:
    fake, speaker = pipelined
    fake.hold()
    speaker.enqueue("Long monologue.")
    speaker.enqueue("Queued behind it.")
    assert _wait_for(lambda: fake.playing == "Long monologue.")
    speaker.close(timeout=_TIMEOUT)  # provider.stop() unblocks playback
    assert fake.stop_calls >= 1
    assert speaker._worker.is_alive() is False
    assert speaker._prefetch is not None
    assert speaker._prefetch.is_alive() is False
    assert "Queued behind it." not in fake.ops("play_start")


def test_provider_with_partial_capability_uses_speak() -> None:
    # Only BOTH synthesize() and play_prepared() enable the pipeline; a
    # provider with just one of them keeps the serial speak() path.
    fake = FakeTTS()
    fake.synthesize = lambda text: ("prepared", text)  # type: ignore[attr-defined]
    speaker = Speaker(fake)
    try:
        assert speaker._prefetch is None
        speaker.enqueue("Spoken serially.")
        assert speaker.wait_until_done(timeout=_TIMEOUT)
        assert fake.spoken == ["Spoken serially."]
    finally:
        speaker.close(timeout=_TIMEOUT)
