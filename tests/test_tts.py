"""Tests for TTS provider selection, incl. the local Kokoro neural voice.

No audio is played and the ~310 MB Kokoro model is never loaded — the real
KokoroTTS class is monkeypatched out where a positive selection is checked,
and the synthesize/play_prepared split is exercised against a fake model and
a fake sounddevice output stream.
"""

from __future__ import annotations

import pytest


def test_create_tts_say_default(config):
    from ada.tts import create_tts
    from ada.tts.say import SayTTS

    config.tts.provider = "say"
    assert isinstance(create_tts(config), SayTTS)


def test_create_tts_unknown_raises(config):
    from ada.tts import create_tts

    config.tts.provider = "bogus"
    with pytest.raises(ValueError):
        create_tts(config)


def test_create_tts_kokoro_falls_back_when_model_missing(config, monkeypatch):
    import ada.tts.kokoro as kmod
    from ada.tts import create_tts
    from ada.tts.say import SayTTS

    monkeypatch.setattr(kmod, "models_present", lambda: False)
    config.tts.provider = "kokoro"
    # No downloaded model -> graceful fallback to the macOS voice.
    assert isinstance(create_tts(config), SayTTS)


def test_create_tts_kokoro_selected(config, monkeypatch):
    import ada.tts.kokoro as kmod
    from ada.tts import create_tts

    monkeypatch.setattr(kmod, "models_present", lambda: True)

    made = {}

    class _FakeKokoro:
        name = "kokoro"

        def __init__(self, voice, speed, lang):
            made["args"] = (voice, speed, lang)

    monkeypatch.setattr(kmod, "KokoroTTS", _FakeKokoro)
    config.tts.provider = "kokoro"
    config.tts.kokoro.voice = "am_michael"
    config.tts.kokoro.speed = 1.1
    config.tts.kokoro.lang = ""  # empty -> None (auto)

    provider = create_tts(config)
    assert provider.name == "kokoro"
    assert made["args"] == ("am_michael", 1.1, None)


def test_kokoro_lang_derivation():
    from ada.tts.kokoro import _lang_for

    assert _lang_for("af_heart", None) == "en-us"
    assert _lang_for("bf_emma", None) == "en-gb"
    assert _lang_for("bm_george", None) == "en-gb"
    assert _lang_for("af_heart", "en-gb") == "en-gb"  # explicit override wins


def test_kokoro_models_present_reflects_files(monkeypatch, tmp_path):
    import ada.tts.kokoro as kmod

    monkeypatch.setattr(kmod, "model_dir", lambda: tmp_path)
    assert kmod.models_present() is False  # empty dir

    model, voices = kmod.model_paths()
    model.write_bytes(b"\0" * 60_000_000)   # plausible model size
    voices.write_bytes(b"\0" * 2_000_000)   # plausible voices size
    assert kmod.models_present() is True


# -- synthesize / play_prepared split (Speaker pipelining capability) ----------
# The real model is never loaded: _preload is a no-op and _load returns a fake.


@pytest.fixture
def kokoro(monkeypatch):
    from ada.tts.kokoro import KokoroTTS

    monkeypatch.setattr(KokoroTTS, "_preload", lambda self: None)
    tts = KokoroTTS(voice="af_heart")

    class _FakeModel:
        def create(self, text, voice, speed, lang):
            assert (voice, speed, lang) == ("af_heart", 1.0, "en-us")
            return [0.0, 0.5, -0.5, 2.0], 24000  # 2.0 exercises clipping

    monkeypatch.setattr(tts, "_load", lambda: _FakeModel())
    return tts


class _FakeStream:
    """Stand-in for sounddevice.RawOutputStream recording chunked writes."""

    def __init__(self, samplerate, channels, dtype):
        assert (samplerate, channels, dtype) == (24000, 1, "int16")
        self.writes: list[int] = []
        self.finalized: list[str] = []
        self.on_write = None  # optional hook, called after each write

    def start(self):
        self.finalized.append("start")

    def write(self, data):
        self.writes.append(len(data))
        if self.on_write is not None:
            self.on_write()

    def stop(self):
        self.finalized.append("stop")

    def abort(self):
        self.finalized.append("abort")

    def close(self):
        self.finalized.append("close")


@pytest.fixture
def fake_stream(monkeypatch):
    import ada.tts.kokoro as kmod

    made: list[_FakeStream] = []

    def factory(**kwargs):
        stream = _FakeStream(**kwargs)
        made.append(stream)
        return stream

    monkeypatch.setattr(kmod.sd, "RawOutputStream", factory)
    return made


def test_kokoro_synthesize_returns_prepared_pcm(kokoro):
    import numpy as np

    prepared = kokoro.synthesize("Hello there.")
    assert prepared.dtype == np.int16
    # Scaled to int16 and clipped (2.0 -> 1.0 -> 32767), no audio I/O yet.
    assert prepared.tolist() == [0, 16383, -16383, 32767]
    # Blank text is prepared as "nothing to play".
    assert kokoro.synthesize("") is None
    assert kokoro.synthesize("   ") is None


def test_kokoro_play_prepared_streams_in_chunks(kokoro, fake_stream):
    import numpy as np

    kokoro.arm()
    pcm = np.zeros(24000, dtype=np.int16)  # 1 s of audio = 48000 bytes
    kokoro.play_prepared(pcm)
    assert len(fake_stream) == 1
    stream = fake_stream[0]
    # ~100 ms chunks so a stop() lands within one chunk: 10 x 4800 bytes.
    assert stream.writes == [4800] * 10
    assert stream.finalized == ["start", "stop", "close"]


def test_kokoro_play_prepared_skips_when_already_stopped(kokoro, fake_stream):
    import numpy as np

    kokoro.arm()
    kokoro.stop()  # barge-in landed between synthesis and playback
    kokoro.play_prepared(np.zeros(24000, dtype=np.int16))
    kokoro.play_prepared(None)  # blank synthesis result is a no-op too
    assert fake_stream == []  # no stream was ever opened


def test_kokoro_play_prepared_honors_stop_mid_playback(kokoro, monkeypatch):
    import numpy as np

    import ada.tts.kokoro as kmod

    made: list[_FakeStream] = []

    def factory(**kwargs):
        stream = _FakeStream(**kwargs)
        # Interrupt after the first chunk, as a concurrent stop() would.
        stream.on_write = lambda: kokoro._interrupt.set()
        made.append(stream)
        return stream

    monkeypatch.setattr(kmod.sd, "RawOutputStream", factory)
    kokoro.arm()
    kokoro.play_prepared(np.zeros(24000, dtype=np.int16))
    (stream,) = made
    assert stream.writes == [4800]  # stopped after one ~100 ms chunk
    assert "abort" in stream.finalized and "close" in stream.finalized


def test_kokoro_speak_composes_synthesize_and_play(kokoro, monkeypatch):
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        kokoro, "synthesize", lambda text: calls.append(("synth", text)) or "prepared"
    )
    monkeypatch.setattr(
        kokoro, "play_prepared", lambda prepared: calls.append(("play", prepared))
    )
    kokoro.speak("Hello there.")
    assert calls == [("synth", "Hello there."), ("play", "prepared")]


def test_kokoro_speak_swallows_synthesis_errors(kokoro, monkeypatch):
    def boom(text):
        raise RuntimeError("synthetic synthesis failure")

    played: list[object] = []
    monkeypatch.setattr(kokoro, "synthesize", boom)
    monkeypatch.setattr(kokoro, "play_prepared", played.append)
    kokoro.speak("Hello there.")  # must not raise (worker-thread contract)
    assert played == []
