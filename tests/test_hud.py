"""Tests for the HUD-facing plumbing.

Covers the UI event bus, the shared answer path (`_answer` / `submit_text`),
and the JS bridge object — all without launching a real window. No pywebview
window is created, no model or mic is touched (the brain and speaker are
faked).
"""

from __future__ import annotations

import threading

import pytest

from ada.ui import events as ui_events


@pytest.fixture(autouse=True)
def _clean_bus():
    ui_events.reset()
    yield
    ui_events.reset()


# -- event bus ---------------------------------------------------------------
def test_event_bus_pub_sub_unsub():
    got = []
    fn = lambda k, p: got.append((k, p))  # noqa: E731
    ui_events.subscribe(fn)
    ui_events.publish("user_text", "hi")
    ui_events.publish("assistant_delta", "yo")
    assert got == [("user_text", "hi"), ("assistant_delta", "yo")]
    ui_events.unsubscribe(fn)
    ui_events.publish("user_text", "bye")
    assert len(got) == 2  # no longer receiving


def test_event_bus_isolates_listener_errors():
    good = []

    def boom(_k, _p):
        raise RuntimeError("listener blew up")

    ui_events.subscribe(boom)
    ui_events.subscribe(lambda _k, p: good.append(p))
    ui_events.publish("x", "ok")  # must not raise
    assert good == ["ok"]


# -- shared answer path ------------------------------------------------------
class _FakeBrain:
    def handle(self, text, on_sentence, cancel=None):
        for s in ("Hello there.", "How can I help?"):
            on_sentence(s)
        return "Hello there. How can I help?"


class _FakeSpeaker:
    def __init__(self):
        self.enqueued = []
        self._gen = 7

    def current_generation(self):
        return self._gen

    def enqueue(self, text, generation=None):
        self.enqueued.append((text, generation))

    def wait_until_done(self, timeout=None):
        return True

    def stop(self):
        pass


def _assistant(config, brain=None):
    from ada.main import Assistant
    from ada.safety import AuditLog
    from ada.safety.confirm import ConsoleConfirmer
    from ada.state import AssistantState

    state = AssistantState()
    a = Assistant(config, state, console=None)
    a._confirmer = ConsoleConfirmer()
    a._audit = AuditLog(config.log_dir / "t.jsonl", enabled=False)
    a.brain = brain or _FakeBrain()
    a.speaker = _FakeSpeaker()
    return a, state


def test_answer_publishes_transcript_and_speaks(config):
    from ada.state import Status

    a, state = _assistant(config)
    events = []
    ui_events.subscribe(lambda k, p: events.append((k, p)))

    a._answer("say hi to me", threading.Event())

    kinds = [k for k, _ in events]
    assert kinds[0] == "user_text"
    assert ("assistant_delta", "Hello there.") in events
    assert ("assistant_delta", "How can I help?") in events
    assert any(k == "assistant_done" for k, _ in events)
    # spoken in order, each tagged with the turn's speaker generation
    assert [t for t, _ in a.speaker.enqueued] == ["Hello there.", "How can I help?"]
    assert all(g == 7 for _, g in a.speaker.enqueued)
    assert state.status is Status.IDLE  # back at rest


def test_answer_skips_brain_when_cancelled(config):
    class _Boom:
        def handle(self, *a, **k):
            raise AssertionError("brain must not run when cancelled")

    a, _state = _assistant(config, brain=_Boom())
    events = []
    ui_events.subscribe(lambda k, p: events.append((k, p)))

    cancel = threading.Event()
    cancel.set()
    a._answer("hello", cancel)

    assert ("user_text", "hello") in events
    assert not any(k == "assistant_delta" for k, _ in events)


def test_submit_text_runs_on_a_thread(config):
    a, _state = _assistant(config)
    events = []
    ui_events.subscribe(lambda k, p: events.append((k, p)))

    a.submit_text("hello")
    turn = a._turn_thread
    assert turn is not None
    turn.join(timeout=5)
    assert not turn.is_alive()
    assert any(k == "assistant_done" for k, _ in events)


def test_submit_text_ignores_blank(config):
    a, _state = _assistant(config)
    a.submit_text("   ")
    assert a._turn_thread is None


# -- JS bridge ---------------------------------------------------------------
def test_ada_api_bridges_to_assistant(config):
    from ada.state import AssistantState
    from ada.ui.hud.app import _AdaApi

    state = AssistantState()
    calls = {"trigger": 0, "text": None}

    class _FakeAssistant:
        def trigger(self):
            calls["trigger"] += 1

        def submit_text(self, t):
            calls["text"] = t

    api = _AdaApi(state, _FakeAssistant(), config)

    api.talk()
    assert calls["trigger"] == 1

    api.sendText("hi ada")
    assert calls["text"] == "hi ada"

    before = state.muted
    api.toggleMute()
    assert state.muted is not before

    # Pushing to JS without a bound window / before ready must never raise.
    api.push_state(state.status, "")
    api.push_event("assistant_delta", "x")

    api.quit()
    assert state.shutting_down


def test_hud_available_and_html_loads():
    from ada.ui.hud import hud_available
    from ada.ui.hud.app import _load_html

    assert hud_available() is True
    html = _load_html()
    assert 'id="orb"' in html and "window.ada" in html
    # the rich JS API surface the bridge pushes into
    for method in ("display(", "tool(", "timing(", "configure("):
        assert method in html
    assert len(html) > 10000


# -- event forwarding into the webview ----------------------------------------
class _FakeWindow:
    """Captures every evaluate_js payload instead of touching a real webview."""

    def __init__(self):
        self.scripts: list[str] = []

    def evaluate_js(self, js):
        self.scripts.append(js)


def _ready_api(config):
    """An _AdaApi wired to a fake window, past the ready() handshake."""
    import json

    from ada.state import AssistantState
    from ada.ui.hud.app import _AdaApi

    api = _AdaApi(AssistantState(), object(), config)
    window = _FakeWindow()
    api.bind_window(window)
    api.ready()
    return api, window, json


def _calls(window, method):
    """The decoded JSON payloads of every `window.ada.<method>(...)` push."""
    import json

    prefix = f"window.ada.{method}("
    return [
        json.loads(js[len(prefix):-1])
        for js in window.scripts
        if js.startswith(prefix)
    ]


def test_push_event_routes_display_tool_timing(config):
    api, window, _json = _ready_api(config)
    window.scripts.clear()

    card = {"kind": "timer", "id": "t1", "label": "Tea", "ends_at": 1.5, "duration_s": 60}
    tool = {"name": "open_app", "summary": "Opening Safari…", "state": "running"}
    timing = {"stages": [["stt", 480.0], ["brain", 1200.5]], "total_ms": 1680.5}

    api.push_event("display", card)
    api.push_event("tool", tool)
    api.push_event("timing", timing)

    assert _calls(window, "display") == [card]
    assert _calls(window, "tool") == [tool]
    assert _calls(window, "timing") == [timing]


def test_push_event_timing_suppressed_when_configured_off(config):
    config.ui.show_timings = False
    api, window, _json = _ready_api(config)
    window.scripts.clear()

    api.push_event("timing", {"stages": [["stt", 100]], "total_ms": 100})
    assert _calls(window, "timing") == []
    # other rich events still flow
    api.push_event("display", {"kind": "markdown", "body": "hi"})
    assert _calls(window, "display") == [{"kind": "markdown", "body": "hi"}]


def test_push_event_drops_unserializable_payload(config):
    api, window, _json = _ready_api(config)
    window.scripts.clear()

    api.push_event("display", {"kind": "table", "rows": object()})  # must not raise
    assert window.scripts == []


def test_configure_pushed_on_ready(config):
    _api, window, _json = _ready_api(config)

    configured = _calls(window, "configure")
    assert configured == [{"showTimings": True, "name": "Ada"}]
    # ready() also pushes the state + mute snapshot after configure
    assert any(js.startswith("window.ada.setState(") for js in window.scripts)
    assert any(js.startswith("window.ada.setMuted(") for js in window.scripts)


# -- openUrl (card/markdown links) ---------------------------------------------
def test_open_url_allows_only_http_https(config, monkeypatch):
    import ada.ui.hud.app as hud_app

    opened = []
    monkeypatch.setattr(hud_app.webbrowser, "open", lambda u: opened.append(u))
    api, _window, _json = _ready_api(config)

    api.openUrl("https://example.com/page")
    api.openUrl("http://example.com")
    assert opened == ["https://example.com/page", "http://example.com"]

    for bad in ("file:///etc/passwd", "javascript:alert(1)", "ftp://x", "", None, 42):
        api.openUrl(bad)
    assert len(opened) == 2  # nothing new opened


def test_open_url_survives_browser_failure(config, monkeypatch):
    import ada.ui.hud.app as hud_app

    def boom(_u):
        raise RuntimeError("no browser")

    monkeypatch.setattr(hud_app.webbrowser, "open", boom)
    api, _window, _json = _ready_api(config)
    api.openUrl("https://example.com")  # must not raise across the JS bridge


# -- window construction helpers ------------------------------------------------
def test_accepts_kwarg_detects_signature():
    from ada.ui.hud.app import _accepts_kwarg

    def with_vibrancy(title, vibrancy=False):
        pass

    def without(title):
        pass

    assert _accepts_kwarg(with_vibrancy, "vibrancy") is True
    assert _accepts_kwarg(without, "vibrancy") is False
    assert _accepts_kwarg(object(), "vibrancy") is False  # unsignaturable
