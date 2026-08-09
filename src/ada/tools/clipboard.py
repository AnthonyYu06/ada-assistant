"""Clipboard tools: read and set the macOS pasteboard via pbpaste/pbcopy."""

from __future__ import annotations

import logging
import subprocess

from .registry import ToolError, ToolResult, tool

log = logging.getLogger("ada.tools.clipboard")

_SPEECH_CHARS = 80
_CARD_CHARS = 4000
_TIMEOUT_S = 5


@tool(
    name="read_clipboard",
    description=(
        "Read the current text on the macOS clipboard. The clipboard may "
        "contain sensitive data (passwords, private messages) — only use "
        "this when the user explicitly asks about their clipboard or what "
        "they copied."
    ),
    input_schema={"type": "object", "properties": {}},
    risk="safe",
    category="clipboard",
)
def read_clipboard() -> str | ToolResult:
    try:
        proc = subprocess.run(
            ["/usr/bin/pbpaste"], capture_output=True, text=True, timeout=_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        raise ToolError("Reading the clipboard timed out.") from None
    if proc.returncode != 0:
        log.debug("pbpaste failed: %s", (proc.stderr or "").strip())
        raise ToolError("I couldn't read the clipboard.")
    text = proc.stdout
    if not text.strip():
        return "The clipboard is empty."

    spoken = " ".join(text.split())
    if len(spoken) > _SPEECH_CHARS:
        spoken = spoken[:_SPEECH_CHARS].rstrip() + "…"
    body = text.strip()
    if len(body) > _CARD_CHARS:
        body = body[:_CARD_CHARS].rstrip() + "…"
    return ToolResult(
        speech=spoken,
        display={"kind": "markdown", "title": "Clipboard", "body": body},
    )


@tool(
    name="set_clipboard",
    description=(
        "Put text on the macOS clipboard, replacing what's there, so the "
        "user can paste it. Use when the user asks to copy something."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to copy."}
        },
        "required": ["text"],
    },
    risk="safe",
    category="clipboard",
)
def set_clipboard(text: str) -> str:
    if not isinstance(text, str) or not text:
        raise ToolError("Tell me what to copy.")
    try:
        proc = subprocess.run(
            ["/usr/bin/pbcopy"], input=text, text=True,
            capture_output=True, timeout=_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise ToolError("Copying to the clipboard timed out.") from None
    if proc.returncode != 0:
        log.debug("pbcopy failed: %s", (proc.stderr or "").strip())
        raise ToolError("I couldn't write to the clipboard.")
    return "Copied."
