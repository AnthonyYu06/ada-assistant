"""Text input tool: type into the frontmost app via System Events.

This is deliberately the ONLY raw-input tool (no mouse control, no arbitrary
key chords). Its safety design is the confirmation dialog: the tool is
risk="confirm" and describe_call shows the exact text that will be typed, so
the user always sees and approves what lands in the front window.
"""

from __future__ import annotations

import logging

from .applescript import applescript_quote, run_applescript
from .registry import ToolError, tool

log = logging.getLogger("ada.tools.input")

_MAX_TEXT_CHARS = 1000
_PREVIEW_CHARS = 140


def _describe_type_text(args: dict) -> str:
    text = str(args.get("text", ""))
    shown = text if len(text) <= _PREVIEW_CHARS else text[:_PREVIEW_CHARS] + "…"
    return f'Type into the front app: "{shown}"'


@tool(
    name="type_text",
    description=(
        "Type text into the frontmost app at the current cursor position, as "
        "if the user typed it. Use when the user asks you to type, dictate, "
        "or fill something in for them. Make sure the right app and field "
        "are focused first."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The exact text to type (newlines press Return).",
            }
        },
        "required": ["text"],
    },
    risk="confirm",
    category="input",
    describe_call=_describe_type_text,
)
def type_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        raise ToolError("Tell me what to type.")
    if len(text) > _MAX_TEXT_CHARS:
        raise ToolError(
            "That's too much text to type — keep it under a thousand characters."
        )
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    body: list[str] = []
    for i, line in enumerate(lines):
        if i > 0:
            body.append("\tkeystroke return")
        if line:
            body.append(f"\tkeystroke {applescript_quote(line)}")
    script = 'tell application "System Events"\n' + "\n".join(body) + "\nend tell"
    run_applescript(script, timeout=15.0)
    return "Typed."
