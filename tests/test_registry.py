"""Tests for ada.tools.registry: @tool, specs, Anthropic shapes, ToolError."""

from __future__ import annotations

import pytest

from ada.tools import registry
from ada.tools.registry import ToolError, ToolSpec, tool


_SCHEMA = {
    "type": "object",
    "properties": {"target": {"type": "string"}},
    "required": ["target"],
}


def test_tool_decorator_registers_spec(clean_registry) -> None:
    @tool(
        name="wave",
        description="Wave at something.",
        input_schema=_SCHEMA,
        risk="confirm",
        category="misc",
    )
    def wave(target: str) -> str:
        return f"waved at {target}"

    spec = registry.get("wave")
    assert spec is not None
    assert spec.name == "wave"
    assert spec.description == "Wave at something."
    assert spec.input_schema == _SCHEMA
    assert spec.risk == "confirm"
    assert spec.category == "misc"
    assert spec.handler is wave
    assert spec.handler(target="moon") == "waved at moon"
    # The decorator returns the original function unchanged.
    assert wave(target="sun") == "waved at sun"


def test_duplicate_name_raises(clean_registry) -> None:
    @tool(name="dupe", description="First.", input_schema={"type": "object"})
    def first() -> str:
        return "one"

    with pytest.raises(ValueError, match="dupe"):
        @tool(name="dupe", description="Second.", input_schema={"type": "object"})
        def second() -> str:
            return "two"

    # The original registration is untouched.
    spec = registry.get("dupe")
    assert spec is not None
    assert spec.handler() == "one"


def test_get_unknown_returns_none(clean_registry) -> None:
    assert registry.get("no-such-tool") is None


def test_anthropic_tools_shape_and_order(clean_registry) -> None:
    @tool(name="zeta", description="Z tool.", input_schema=_SCHEMA)
    def zeta(target: str) -> str:
        return ""

    @tool(name="alpha", description="A tool.", input_schema={"type": "object"})
    def alpha() -> str:
        return ""

    tools = registry.anthropic_tools()
    assert [t["name"] for t in tools] == ["alpha", "zeta"]  # sorted by name
    for entry in tools:
        # Exactly the keys the Anthropic Messages API expects — no extras
        # like handler/risk leaking through.
        assert set(entry) == {"name", "description", "input_schema"}
    assert tools[1]["input_schema"] == _SCHEMA
    assert tools[0]["description"] == "A tool."


def test_all_tools_sorted(clean_registry) -> None:
    for name in ("banana", "apple", "cherry"):
        tool(name=name, description=name, input_schema={"type": "object"})(
            lambda: ""
        )
    assert [s.name for s in registry.all_tools()] == ["apple", "banana", "cherry"]


def test_summary_with_describe_call(clean_registry) -> None:
    spec = ToolSpec(
        name="open_app",
        description="Open an app.",
        input_schema=_SCHEMA,
        handler=lambda **kw: "",
        describe_call=lambda a: f"Open {a['app_name']}",
    )
    assert spec.summary({"app_name": "Safari"}) == "Open Safari"


def test_summary_falls_back_when_describe_call_fails(clean_registry) -> None:
    spec = ToolSpec(
        name="open_app",
        description="Open an app.",
        input_schema=_SCHEMA,
        handler=lambda **kw: "",
        describe_call=lambda a: f"Open {a['missing_key']}",  # raises KeyError
    )
    assert spec.summary({"app_name": "Safari"}) == "open_app(app_name='Safari')"


def test_summary_without_describe_call(clean_registry) -> None:
    spec = ToolSpec(
        name="set_timer",
        description="Set a timer.",
        input_schema={"type": "object"},
        handler=lambda **kw: "",
    )
    assert spec.summary({"duration_seconds": 60, "label": "tea"}) == (
        "set_timer(duration_seconds=60, label='tea')"
    )


def test_clear_empties_registry(clean_registry) -> None:
    tool(name="temp", description="t", input_schema={"type": "object"})(lambda: "")
    assert registry.get("temp") is not None
    registry.clear()
    assert registry.get("temp") is None
    assert registry.all_tools() == []
    assert registry.anthropic_tools() == []


def test_tool_error_message() -> None:
    err = ToolError("The printer is on fire.")
    assert isinstance(err, Exception)
    assert str(err) == "The printer is on fire."
    with pytest.raises(ToolError, match="on fire"):
        raise ToolError("The printer is on fire.")
