"""Display tools: put rich content on the user's screen via HUD cards."""

from __future__ import annotations

import logging

from .registry import ToolError, ToolResult, tool

log = logging.getLogger("ada.tools.display")

_CHART_KINDS = ("bar", "line")
_MAX_POINTS = 50
_MAX_BODY_CHARS = 8000


@tool(
    name="show_on_screen",
    description=(
        "Show text, lists, or structured information on the user's screen as "
        "a card. Use this whenever an answer has structure (lists, steps, "
        "comparisons, code, long text) and then speak only a short summary. "
        "The body supports mini-markdown: #/## headings, **bold**, *italic*, "
        "`code`, - bullets, 1. numbered lists, and [text](url) links."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short card title, e.g. 'Packing list'.",
            },
            "body": {
                "type": "string",
                "description": "The content to display, in mini-markdown.",
            },
        },
        "required": ["title", "body"],
    },
    risk="safe",
    category="display",
)
def show_on_screen(title: str, body: str) -> ToolResult:
    title = (title or "").strip()
    if not body or not body.strip():
        raise ToolError("Tell me what to show on screen.")
    body = body.strip()
    if len(body) > _MAX_BODY_CHARS:
        body = body[:_MAX_BODY_CHARS].rstrip() + "…"
    return ToolResult(
        speech="It's on your screen.",
        display={"kind": "markdown", "title": title, "body": body},
    )


@tool(
    name="show_chart",
    description=(
        "Draw a bar or line chart on the user's screen. Use whenever an "
        "answer involves numbers over categories or time (prices, "
        "temperatures, counts, trends) — show the chart, then speak only a "
        "one-sentence takeaway."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short chart title, e.g. 'Weekly temperatures'.",
            },
            "chart": {
                "type": "string",
                "enum": list(_CHART_KINDS),
                "description": "'bar' for categories, 'line' for trends over time.",
            },
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One label per data point, e.g. days or category names.",
            },
            "values": {
                "type": "array",
                "items": {"type": "number"},
                "description": "The numeric values, one per label.",
            },
            "unit": {
                "type": "string",
                "description": "Optional unit for the values, e.g. '°F' or '$'.",
            },
        },
        "required": ["title", "chart", "labels", "values"],
    },
    risk="safe",
    category="display",
)
def show_chart(
    title: str,
    chart: str,
    labels: list[str],
    values: list[float],
    unit: str | None = None,
) -> ToolResult:
    if chart not in _CHART_KINDS:
        raise ToolError("The chart type must be 'bar' or 'line'.")
    if not isinstance(labels, list) or not isinstance(values, list):
        raise ToolError("Labels and values must both be lists.")
    if not labels or not values:
        raise ToolError("The chart needs at least one data point.")
    if len(labels) != len(values):
        raise ToolError("The chart needs exactly one value per label.")
    if len(values) > _MAX_POINTS:
        raise ToolError(f"That's too many points to chart — keep it under {_MAX_POINTS}.")
    clean_values: list[float] = []
    for v in values:
        if isinstance(v, bool):
            raise ToolError("Chart values must be numbers.")
        try:
            clean_values.append(float(v))
        except (TypeError, ValueError):
            raise ToolError("Chart values must be numbers.") from None
    card: dict = {
        "kind": "chart",
        "title": (title or "").strip(),
        "chart": chart,
        "labels": [str(l) for l in labels],
        "values": clean_values,
    }
    if unit:
        card["unit"] = str(unit)
    return ToolResult(speech="The chart is on your screen.", display=card)
