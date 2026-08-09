"""File search tools (Spotlight-backed, restricted to allowed roots)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from . import get_config
from .registry import ToolError, ToolResult, tool

log = logging.getLogger("ada.tools.files")


def _abbreviate_home(path: str) -> str:
    home = str(Path.home())
    if path == home:
        return "~"
    if path.startswith(home + "/"):
        return "~" + path[len(home):]
    return path


@tool(
    name="search_files",
    description=(
        "Search the user's allowed folders (by default Documents, Desktop and "
        "Downloads) for files, using Spotlight. Search by file name or by file "
        "content. Use when the user asks to find, locate, or look for a file; "
        "follow up with open_path to open a result."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for — a file name fragment or content keywords.",
            },
            "search_type": {
                "type": "string",
                "enum": ["name", "content"],
                "description": "'name' matches file names; 'content' matches text inside files.",
                "default": "name",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (1-25).",
                "default": 10,
            },
        },
        "required": ["query"],
    },
    risk="safe",
    category="files",
)
def search_files(query: str, search_type: str = "name", limit: int = 10) -> str | ToolResult:
    if search_type not in ("name", "content"):
        raise ToolError("search_type must be 'name' or 'content'.")
    if not query or not query.strip():
        raise ToolError("Tell me what to search for.")
    query = query.strip()
    try:
        limit = max(1, min(25, int(limit)))
    except (TypeError, ValueError):
        limit = 10

    roots = [r for r in get_config().permissions.file_roots_resolved() if r.exists()]

    # Build an explicit metadata query expression rather than a bare argument,
    # so a query starting with '-' isn't parsed by mdfind as an option.
    escaped = query.replace("\\", "\\\\").replace('"', '\\"')
    if search_type == "name":
        query_expr = f'kMDItemFSName == "*{escaped}*"cd'
    else:
        query_expr = f'kMDItemTextContent == "*{escaped}*"cd'

    found: list[str] = []
    seen: set[str] = set()
    for root in roots:
        cmd = ["/usr/bin/mdfind", "-onlyin", str(root), query_expr]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            raise ToolError("File search timed out.") from None
        if proc.returncode != 0:
            log.debug("mdfind failed in %s: %s", root, (proc.stderr or "").strip())
            continue
        for line in proc.stdout.splitlines():
            path = line.strip()
            if path and path not in seen:
                seen.add(path)
                found.append(path)

    found = found[:limit]
    if not found:
        return "No matching files found in your allowed folders."
    noun = "file" if len(found) == 1 else "files"
    lines = [f"Found {len(found)} {noun}."]
    lines.extend(_abbreviate_home(p) for p in found)
    return ToolResult(
        speech="\n".join(lines),
        display={
            "kind": "files",
            "title": f"Files matching '{query}'",
            "items": [
                {"name": Path(p).name, "path": _abbreviate_home(p)} for p in found
            ],
        },
    )
