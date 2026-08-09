"""System control tools: volume, screen lock, sleep, battery, System Settings."""

from __future__ import annotations

import ctypes
import logging
import re
import subprocess

from .applescript import run_applescript
from .registry import ToolError, ToolResult, tool

log = logging.getLogger("ada.tools.system")

# Well-known System Settings pane URLs (macOS 13+ Settings app).
_PANE_URLS: dict[str, str] = {
    "sound": "x-apple.systempreferences:com.apple.Sound-Settings.extension",
    "bluetooth": "x-apple.systempreferences:com.apple.BluetoothSettings",
    "network": "x-apple.systempreferences:com.apple.Network-Settings.extension",
    "displays": "x-apple.systempreferences:com.apple.Displays-Settings.extension",
    "notifications": "x-apple.systempreferences:com.apple.Notifications-Settings.extension",
    "privacy": "x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension",
    "battery": "x-apple.systempreferences:com.apple.Battery-Settings.extension",
    "keyboard": "x-apple.systempreferences:com.apple.Keyboard-Settings.extension",
    "appearance": "x-apple.systempreferences:com.apple.Appearance-Settings.extension",
}


@tool(
    name="set_volume",
    description=(
        "Set the Mac's output volume to a level from 0 (silent) to 100 (max). "
        "Use when the user asks to turn the volume up, down, or to a level."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "level": {
                "type": "integer",
                "description": "Output volume from 0 to 100.",
            }
        },
        "required": ["level"],
    },
    risk="safe",
    category="system",
)
def set_volume(level: int) -> str:
    try:
        level = max(0, min(100, int(level)))
    except (TypeError, ValueError):
        raise ToolError("Volume must be a number from 0 to 100.") from None
    run_applescript(f"set volume output volume {level}")
    return f"Volume set to {level} percent."


@tool(
    name="mute_audio",
    description="Mute or unmute the Mac's audio output.",
    input_schema={
        "type": "object",
        "properties": {
            "mute": {
                "type": "boolean",
                "description": "true to mute, false to unmute.",
            }
        },
        "required": ["mute"],
    },
    risk="safe",
    category="system",
)
def mute_audio(mute: bool) -> str:
    run_applescript(f"set volume output muted {'true' if mute else 'false'}")
    return "Audio muted." if mute else "Audio unmuted."


@tool(
    name="lock_screen",
    description=(
        "Lock the Mac by putting the display to sleep. The user will need to "
        "log back in. Use when asked to lock the screen or the computer."
    ),
    input_schema={"type": "object", "properties": {}},
    risk="safe",
    category="system",
)
def lock_screen() -> str:
    # Lock unconditionally via the private login framework. Unlike
    # `pmset displaysleepnow`, this locks even when "require password
    # immediately" is not set.
    try:
        login = ctypes.CDLL(
            "/System/Library/PrivateFrameworks/login.framework/login"
        )
        login.SACLockScreenImmediate()
        return "Locking your screen."
    except (OSError, AttributeError) as exc:
        log.debug("SACLockScreenImmediate failed: %s", exc)

    proc = subprocess.run(
        ["/usr/bin/pmset", "displaysleepnow"], capture_output=True, text=True
    )
    if proc.returncode != 0:
        log.debug("pmset displaysleepnow failed: %s", (proc.stderr or "").strip())
        raise ToolError("I couldn't lock the screen.")
    return "Locking your screen."


@tool(
    name="system_sleep",
    description=(
        "Put the whole Mac to sleep immediately. Use only when the user "
        "explicitly asks to put the computer to sleep."
    ),
    input_schema={"type": "object", "properties": {}},
    risk="confirm",
    category="system",
    describe_call=lambda a: "Put the Mac to sleep",
)
def system_sleep() -> str:
    proc = subprocess.run(
        ["/usr/bin/pmset", "sleepnow"], capture_output=True, text=True
    )
    if proc.returncode != 0:
        log.debug("pmset sleepnow failed: %s", (proc.stderr or "").strip())
        raise ToolError("I couldn't put the Mac to sleep.")
    return "Putting your Mac to sleep."


@tool(
    name="battery_status",
    description=(
        "Report the Mac's battery percentage and whether it is charging. "
        "Use when the user asks about battery level or power state."
    ),
    input_schema={"type": "object", "properties": {}},
    risk="safe",
    category="system",
)
def battery_status() -> str | ToolResult:
    try:
        proc = subprocess.run(
            ["/usr/bin/pmset", "-g", "batt"], capture_output=True, text=True, timeout=10
        )
    except subprocess.TimeoutExpired:
        raise ToolError("That system action timed out.") from None
    if proc.returncode != 0:
        log.debug("pmset -g batt failed: %s", (proc.stderr or "").strip())
        raise ToolError("I couldn't read the battery status.")
    out = proc.stdout
    match = re.search(r"(\d{1,3})%", out)
    if not match:
        return "This Mac doesn't report a battery."
    percent = int(match.group(1))
    lower = out.lower()
    if "discharging" in lower:
        state = "on battery power"
    elif "not charging" in lower:
        state = "plugged in but not charging"
    elif "charged" in lower:
        state = "fully charged"
    elif "charging" in lower:
        state = "charging"
    elif "ac power" in lower:
        state = "plugged in"
    else:
        state = ""
    pairs = [["Battery", f"{percent}%"]]
    if state:
        pairs.append(["State", state])
    speech = (
        f"Battery is at {percent} percent, {state}."
        if state
        else f"Battery is at {percent} percent."
    )
    return ToolResult(
        speech=speech,
        display={"kind": "keyvalue", "title": "Battery", "pairs": pairs},
    )


@tool(
    name="open_settings",
    description=(
        "Open System Settings, optionally jumping straight to a specific pane "
        "such as sound, bluetooth, network, or displays."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "pane": {
                "type": "string",
                "enum": [
                    "general",
                    "appearance",
                    "sound",
                    "bluetooth",
                    "network",
                    "displays",
                    "notifications",
                    "privacy",
                    "battery",
                    "keyboard",
                    "default",
                ],
                "description": "Which settings pane to open; 'default' opens the Settings app itself.",
            }
        },
        "required": ["pane"],
    },
    risk="safe",
    category="system",
)
def open_settings(pane: str) -> str:
    key = (pane or "default").strip().lower()
    url = _PANE_URLS.get(key)
    if url is not None:
        cmd = ["/usr/bin/open", url]
    else:
        cmd = ["/usr/bin/open", "-b", "com.apple.systempreferences"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log.debug("open settings %r failed: %s", key, (proc.stderr or "").strip())
        raise ToolError("I couldn't open System Settings.")
    if url is None:
        return "Opened System Settings."
    return f"Opened {key} settings."
