"""Application and URL/document launching tools."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from .applescript import applescript_quote, run_applescript
from .registry import ToolError, tool

log = logging.getLogger("ada.tools.apps")


@tool(
    name="open_app",
    description=(
        "Open (launch or bring to the front) a macOS application by name, "
        "e.g. Safari, Notes, Music, Terminal. Use when the user asks to open, "
        "launch, start, or switch to an app."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "The application's name as it appears in /Applications, e.g. 'Safari'.",
            }
        },
        "required": ["app_name"],
    },
    risk="safe",
    category="apps",
)
def open_app(app_name: str) -> str:
    proc = subprocess.run(
        ["/usr/bin/open", "-a", app_name], capture_output=True, text=True
    )
    if proc.returncode != 0:
        log.debug("open -a %r failed: %s", app_name, (proc.stderr or "").strip())
        raise ToolError(f"I couldn't find an app called {app_name}.")
    return f"Opened {app_name}."


@tool(
    name="quit_app",
    description=(
        "Quit a running macOS application by name. This may lose unsaved work, "
        "so only use it when the user explicitly asks to quit or close an app."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "The application to quit, e.g. 'Safari'.",
            }
        },
        "required": ["app_name"],
    },
    risk="confirm",
    category="apps",
    describe_call=lambda a: f"Quit {a['app_name']} (unsaved work may be lost)",
)
def quit_app(app_name: str) -> str:
    run_applescript(f"tell application {applescript_quote(app_name)} to quit")
    return f"Quit {app_name}."


@tool(
    name="open_url",
    description=(
        "Open a web URL (http or https only) in the user's default browser. "
        "Use when the user asks to open, visit, or go to a website."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Full URL including the scheme, e.g. 'https://example.com'.",
            }
        },
        "required": ["url"],
    },
    risk="safe",
    category="apps",
)
def open_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ToolError("I can only open http or https links.")
    if not parsed.netloc:
        raise ToolError("That doesn't look like a valid web address.")
    proc = subprocess.run(["/usr/bin/open", url], capture_output=True, text=True)
    if proc.returncode != 0:
        log.debug("open %r failed: %s", url, (proc.stderr or "").strip())
        raise ToolError("I couldn't open that link.")
    return "Opened that in your browser."


@tool(
    name="open_path",
    description=(
        "Open a file or folder on disk with its default application, or reveal "
        "a folder in Finder. Use after search_files to open a result, or when "
        "the user names a specific file or folder."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path or ~-relative path, e.g. '~/Documents/report.pdf'.",
            }
        },
        "required": ["path"],
    },
    risk="safe",
    category="files",
)
def open_path(path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        raise ToolError(f"I can't find {path}.")
    proc = subprocess.run(["/usr/bin/open", str(p)], capture_output=True, text=True)
    if proc.returncode != 0:
        log.debug("open %r failed: %s", str(p), (proc.stderr or "").strip())
        raise ToolError(f"I couldn't open {p.name}.")
    return f"Opened {p.name}."
