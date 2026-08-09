"""Tool registry.

Every capability the assistant can perform is a registered tool with:
  - an Anthropic-compatible JSON schema (name, description, input_schema)
  - a Python handler that returns a short result string for the model
  - a risk level that the safety layer uses to decide on confirmation

Handlers receive the tool input as keyword arguments and must return a
concise, speakable string describing the outcome — or a ToolResult when
they also want something rendered graphically in the HUD. They raise
ToolError for expected failures (bad input, not found, denied) — the
message is returned to the model as an error tool_result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Union

log = logging.getLogger("ada.tools")

RiskLevel = Literal["safe", "confirm"]


@dataclass
class ToolResult:
    """A tool outcome with an optional on-screen card.

    ``speech`` is the short, speakable text fed back to the model (same role
    as a plain-string return). ``display`` is an optional display card for
    the HUD — see the schema in ``ada.ui.events``; it never reaches the
    model, so keep everything the model needs in ``speech``.
    """

    speech: str
    display: dict[str, Any] | None = None


ToolHandler = Callable[..., Union[str, ToolResult]]


class ToolError(Exception):
    """Expected tool failure — message is shown to the model (and user)."""


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    risk: RiskLevel = "safe"
    category: str = "general"
    # One-line, human-readable summary of a call, for confirmation dialogs.
    # Receives the tool args; e.g. lambda a: f"Quit {a['app_name']}".
    describe_call: Callable[[dict[str, Any]], str] | None = None

    def summary(self, args: dict[str, Any]) -> str:
        if self.describe_call is not None:
            try:
                return self.describe_call(args)
            except Exception:  # noqa: BLE001
                pass
        pretty = ", ".join(f"{k}={v!r}" for k, v in args.items())
        return f"{self.name}({pretty})"


_REGISTRY: dict[str, ToolSpec] = {}


def tool(
    name: str,
    description: str,
    input_schema: dict[str, Any],
    risk: RiskLevel = "safe",
    category: str = "general",
    describe_call: Callable[[dict[str, Any]], str] | None = None,
) -> Callable[[ToolHandler], ToolHandler]:
    """Decorator that registers a function as an assistant tool."""

    def decorator(fn: ToolHandler) -> ToolHandler:
        if name in _REGISTRY:
            raise ValueError(f"Duplicate tool name: {name}")
        _REGISTRY[name] = ToolSpec(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=fn,
            risk=risk,
            category=category,
            describe_call=describe_call,
        )
        return fn

    return decorator


def get(name: str) -> ToolSpec | None:
    return _REGISTRY.get(name)


def all_tools() -> list[ToolSpec]:
    return sorted(_REGISTRY.values(), key=lambda spec: spec.name)


def anthropic_tools() -> list[dict[str, Any]]:
    """Tool definitions in the shape the Anthropic Messages API expects."""
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
        }
        for spec in all_tools()
    ]


def clear() -> None:
    """Testing helper — forget all registered tools."""
    _REGISTRY.clear()
