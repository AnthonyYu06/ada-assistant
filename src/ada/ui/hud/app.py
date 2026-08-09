"""The native HUD window (WKWebView via pywebview).

A frameless, translucent, always-on-top liquid-glass panel that floats over
the desktop. It renders the front-end (``index.html``), reflects the
assistant's live state, streams the transcript, renders rich display cards /
tool chips / timing readouts, and lets the user tap-to-talk, type, mute, and
quit.

Threading: pywebview owns the process main thread (``webview.start()`` blocks).
The assistant runs on worker threads and pushes updates here via the UI event
bus and the shared ``AssistantState`` listener; those pushes are marshalled to
the webview with ``window.evaluate_js`` and guarded so a UI hiccup never takes
down the pipeline.
"""

from __future__ import annotations

import inspect
import json
import logging
import webbrowser
from importlib import resources
from typing import Any
from urllib.parse import urlsplit

from ...config import Config
from ...state import AssistantState, Status
from .. import events as ui_events

log = logging.getLogger("ada.ui.hud")

_WIDTH = 430
_HEIGHT = 680
_MIN_WIDTH = 390
_MIN_HEIGHT = 560


def hud_available() -> bool:
    """True when pywebview (and its native backend) can be imported."""
    try:
        import webview  # noqa: F401
        import webview.platforms.cocoa  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _load_html() -> str:
    return resources.files("ada.ui.hud").joinpath("index.html").read_text(encoding="utf-8")


class _AdaApi:
    """The object exposed to JavaScript as ``window.pywebview.api``.

    Every method here is invoked by the front-end (button clicks, Enter, the
    ``pywebviewready`` handshake). Keep them fast and non-blocking — they run
    on pywebview's bridge thread.
    """

    def __init__(self, state: AssistantState, assistant: Any, cfg: Config) -> None:
        self._state = state
        self._assistant = assistant
        self._cfg = cfg
        self._window: Any = None
        self._ready = False

    def bind_window(self, window: Any) -> None:
        self._window = window

    # -- JS -> Python --------------------------------------------------------
    def ready(self) -> None:
        """Front-end finished loading; push config + the current snapshot."""
        self._ready = True
        self._push_json(
            "configure",
            {
                "showTimings": bool(self._cfg.ui.show_timings),
                "name": self._cfg.assistant.name,
            },
        )
        self.push_state(self._state.status, self._state.detail)
        self._eval(f"window.ada.setMuted({json.dumps(self._state.muted)})")

    def openUrl(self, url: Any) -> None:  # noqa: N802 - JS-facing name
        """Open an http/https URL in the default browser (card/markdown links)."""
        try:
            scheme = urlsplit(str(url)).scheme.lower()
        except ValueError:
            scheme = ""
        if scheme not in ("http", "https"):
            log.warning("openUrl refused non-http(s) URL: %.120r", url)
            return
        try:
            webbrowser.open(str(url))
        except Exception:  # noqa: BLE001 - a bad handler must not hit the bridge
            log.exception("openUrl failed")

    def talk(self) -> None:
        try:
            self._assistant.trigger()
        except Exception:  # noqa: BLE001
            log.exception("talk() failed")

    def sendText(self, text: str) -> None:  # noqa: N802 - JS-facing name
        try:
            self._assistant.submit_text(text)
        except Exception:  # noqa: BLE001
            log.exception("sendText() failed")

    def toggleMute(self) -> None:  # noqa: N802 - JS-facing name
        muted = self._state.toggle_muted()
        self._eval(f"window.ada.setMuted({json.dumps(muted)})")

    def quit(self) -> None:
        self._state.request_shutdown()

    # -- Python -> JS (pushes) ----------------------------------------------
    def _eval(self, js: str) -> None:
        window = self._window
        if window is None or not self._ready:
            return
        try:
            window.evaluate_js(js)
        except Exception as exc:  # noqa: BLE001 - webview may be tearing down
            log.debug("evaluate_js failed: %s", exc)

    def push_state(self, status: Status, detail: str = "") -> None:
        value = status.value if isinstance(status, Status) else str(status)
        self._eval(
            f"window.ada.setState({json.dumps(value)}, {json.dumps(detail or '')})"
        )

    def _push_json(self, method: str, payload: Any) -> None:
        """Serialize `payload` and call ``window.ada.<method>(payload)``."""
        try:
            blob = json.dumps(payload)
        except (TypeError, ValueError):
            log.warning("Dropped unserializable %s payload", method)
            return
        self._eval(f"window.ada.{method}({blob})")

    def push_event(self, kind: str, payload: Any) -> None:
        if kind == "user_text":
            self._eval(f"window.ada.userText({json.dumps(str(payload))})")
        elif kind == "assistant_delta":
            self._eval(f"window.ada.assistantDelta({json.dumps(str(payload))})")
        elif kind == "assistant_done":
            self._eval(f"window.ada.assistantDone({json.dumps(str(payload))})")
        elif kind == "level":
            try:
                self._eval(f"window.ada.level({float(payload):.3f})")
            except (TypeError, ValueError):
                pass
        elif kind == "display":
            self._push_json("display", payload)
        elif kind == "tool":
            self._push_json("tool", payload)
        elif kind == "timing":
            if self._cfg.ui.show_timings:
                self._push_json("timing", payload)


def _accepts_kwarg(func: Any, name: str) -> bool:
    """True when `func`'s signature accepts a keyword argument called `name`.

    Used to feature-detect optional pywebview parameters (e.g. ``vibrancy``
    for native macOS blur) without pinning a pywebview version.
    """
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _screen_bottom_right(width: int, height: int) -> tuple[int | None, int | None]:
    """Best-effort bottom-right placement (falls back to centered = None)."""
    try:
        from AppKit import NSScreen

        vf = NSScreen.mainScreen().visibleFrame()
        sw, sh = int(vf.size.width), int(vf.size.height)
        x = max(0, sw - width - 24)
        y = max(0, sh - height - 40)
        return x, y
    except Exception:  # noqa: BLE001
        return None, None


def run_hud(state: AssistantState, assistant: Any, cfg: Config) -> None:
    """Create and run the HUD window. Blocks the main thread until it closes.

    The caller must have already started the assistant (warmup + voice loop)
    on background threads.
    """
    import webview

    api = _AdaApi(state, assistant, cfg)
    x, y = _screen_bottom_right(_WIDTH, _HEIGHT)

    kwargs: dict[str, Any] = dict(
        html=_load_html(),
        js_api=api,
        width=_WIDTH,
        height=_HEIGHT,
        x=x,
        y=y,
        resizable=True,
        frameless=True,
        easy_drag=True,
        on_top=True,
        transparent=True,
        background_color="#0A0E14",
        text_select=True,
        min_size=(_MIN_WIDTH, _MIN_HEIGHT),
    )
    # Native macOS blur behind the glass, when this pywebview supports it.
    if _accepts_kwarg(webview.create_window, "vibrancy"):
        kwargs["vibrancy"] = True
    try:
        window = webview.create_window(cfg.assistant.name, **kwargs)
    except TypeError:
        if "vibrancy" not in kwargs:
            raise
        kwargs.pop("vibrancy")  # signature lied (e.g. **kwargs shim) — retry
        window = webview.create_window(cfg.assistant.name, **kwargs)
    api.bind_window(window)

    # Live wiring: status changes + transcript events push into the webview.
    state.add_listener(api.push_state)
    ui_events.subscribe(api.push_event)

    # Closing the window (or a shutdown from elsewhere) ends the session.
    def _on_closed() -> None:
        state.request_shutdown()

    window.events.closed += _on_closed

    # When shutdown is requested from voice/quit, tear the window down so
    # webview.start() returns and the CLI can clean up.
    def _watch_shutdown(status: Status, _detail: str) -> None:
        if state.shutting_down:
            try:
                window.destroy()
            except Exception:  # noqa: BLE001
                pass

    state.add_listener(_watch_shutdown)

    log.info("HUD window created (%dx%d)", _WIDTH, _HEIGHT)
    try:
        webview.start()
    finally:
        ui_events.unsubscribe(api.push_event)
