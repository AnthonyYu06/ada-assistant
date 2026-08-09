"""Tests for ada.tools.music: player selection, controls, now-playing card.

All AppleScript is mocked at the import site (ada.tools.music.run_applescript).
"""

from __future__ import annotations

import importlib

import pytest

from ada.tools import registry
from ada.tools.registry import ToolError, ToolResult

music = importlib.import_module("ada.tools.music")


def _fake_applescript(
    monkeypatch: pytest.MonkeyPatch, responses: list[object]
) -> list[str]:
    """Queue run_applescript responses; returns the captured scripts."""
    calls: list[str] = []
    queue = list(responses)

    def fake(script: str, timeout: float = 10.0) -> str:
        calls.append(script)
        result = queue.pop(0) if queue else ""
        if isinstance(result, Exception):
            raise result
        return str(result)

    monkeypatch.setattr("ada.tools.music.run_applescript", fake)
    return calls


# -- music_control ------------------------------------------------------------

def test_music_control_prefers_spotify(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_applescript(monkeypatch, ["true|||true", ""])
    assert music.music_control("pause") == "Paused."
    assert 'tell application "Spotify" to pause' in calls[1]


def test_music_control_falls_back_to_music(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_applescript(monkeypatch, ["false|||true", ""])
    assert music.music_control("next") == "Skipped."
    assert 'tell application "Music" to next track' in calls[1]


def test_music_control_toggle_uses_playpause(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_applescript(monkeypatch, ["true|||false", ""])
    music.music_control("toggle")
    assert "playpause" in calls[1]


def test_music_control_no_player(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_applescript(monkeypatch, ["false|||false"])
    with pytest.raises(ToolError, match="No music app"):
        music.music_control("play")


def test_music_control_rejects_unknown_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_applescript(monkeypatch, [])
    with pytest.raises(ToolError, match="don't know"):
        music.music_control("shuffle")
    assert calls == []  # validated before any AppleScript runs


# -- now_playing --------------------------------------------------------------

def test_now_playing_card(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_applescript(
        monkeypatch, ["true|||false", "Halo|||Beyonce|||I Am|||playing"]
    )
    result = music.now_playing()
    assert isinstance(result, ToolResult)
    assert "Halo" in result.speech
    assert "Beyonce" in result.speech
    assert result.display == {
        "kind": "now_playing",
        "track": "Halo",
        "artist": "Beyonce",
        "album": "I Am",
        "state": "playing",
        "app": "Spotify",
    }


def test_now_playing_paused_without_album(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_applescript(monkeypatch, ["false|||true", "Song|||Artist||||||paused"])
    result = music.now_playing()
    assert isinstance(result, ToolResult)
    assert result.speech.startswith("Paused")
    assert result.display["state"] == "paused"
    assert result.display["app"] == "Music"
    assert "album" not in result.display


def test_now_playing_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_applescript(monkeypatch, ["false|||true", "__STOPPED__"])
    assert music.now_playing() == "Nothing is playing right now."


def test_now_playing_no_player(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_applescript(monkeypatch, ["false|||false"])
    with pytest.raises(ToolError, match="No music app"):
        music.now_playing()


def test_now_playing_garbled_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_applescript(monkeypatch, ["true|||false", "nonsense"])
    with pytest.raises(ToolError, match="couldn't read"):
        music.now_playing()


# -- registration -------------------------------------------------------------

def test_music_tools_registered_safe() -> None:
    for name in ("music_control", "now_playing"):
        spec = registry.get(name)
        assert spec is not None
        assert spec.risk == "safe"
        assert spec.category == "music"
