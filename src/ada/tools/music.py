"""Music tools: control and inspect the running player (Spotify or Music)."""

from __future__ import annotations

import logging

from .applescript import run_applescript
from .registry import ToolError, ToolResult, tool

log = logging.getLogger("ada.tools.music")

_SEP = "|||"
_STOPPED = "__STOPPED__"

_PLAYER_CHECK = """\
tell application "System Events"
	set hasSpotify to exists process "Spotify"
	set hasMusic to exists process "Music"
end tell
return (hasSpotify as text) & "|||" & (hasMusic as text)"""

_ACTIONS: dict[str, tuple[str, str]] = {
    # action -> (AppleScript command, spoken confirmation)
    "play": ("play", "Playing."),
    "pause": ("pause", "Paused."),
    "toggle": ("playpause", "Toggled playback."),
    "next": ("next track", "Skipped."),
    "previous": ("previous track", "Playing the previous track."),
}


def _running_player() -> str:
    """Return the app to target ("Spotify" or "Music"), preferring Spotify."""
    out = run_applescript(_PLAYER_CHECK)
    parts = [p.strip().lower() for p in out.split(_SEP)]
    while len(parts) < 2:
        parts.append("")
    if parts[0] == "true":
        return "Spotify"
    if parts[1] == "true":
        return "Music"
    raise ToolError("No music app is open.")


@tool(
    name="music_control",
    description=(
        "Control music playback in the running player (Spotify or Apple "
        "Music): play, pause, toggle play/pause, or skip to the next or "
        "previous track. Use when the user asks to play, pause, resume, or "
        "skip music."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
                "description": "What to do with playback.",
            }
        },
        "required": ["action"],
    },
    risk="safe",
    category="music",
)
def music_control(action: str) -> str:
    if action not in _ACTIONS:
        raise ToolError(f"I don't know the playback action {action}.")
    player = _running_player()
    command, speech = _ACTIONS[action]
    run_applescript(f'tell application "{player}" to {command}')
    return speech


@tool(
    name="now_playing",
    description=(
        "Report the current track, artist, and album from the running music "
        "player (Spotify or Apple Music). Use when the user asks what's "
        "playing or what this song is."
    ),
    input_schema={"type": "object", "properties": {}},
    risk="safe",
    category="music",
)
def now_playing() -> str | ToolResult:
    player = _running_player()
    script = (
        f'tell application "{player}"\n'
        f'\tif player state is stopped then return "{_STOPPED}"\n'
        '\tset trackAlbum to ""\n'
        "\ttry\n"
        "\t\tset trackAlbum to album of current track\n"
        "\tend try\n"
        f'\treturn (name of current track) & "{_SEP}" & '
        f'(artist of current track) & "{_SEP}" & trackAlbum & "{_SEP}" & '
        "(player state as text)\n"
        "end tell"
    )
    out = run_applescript(script)
    if out.strip() == _STOPPED:
        return "Nothing is playing right now."
    parts = out.split(_SEP)
    if len(parts) < 4:
        raise ToolError("I couldn't read what's playing.")
    track, artist, album, state = (p.strip() for p in parts[:4])
    if not track:
        return "Nothing is playing right now."
    state = "paused" if state.lower() == "paused" else "playing"
    card: dict = {
        "kind": "now_playing",
        "track": track,
        "artist": artist,
        "state": state,
        "app": player,
    }
    if album:
        card["album"] = album
    verb = "Now playing" if state == "playing" else "Paused"
    speech = f"{verb}: {track} by {artist}." if artist else f"{verb}: {track}."
    return ToolResult(speech=speech, display=card)
