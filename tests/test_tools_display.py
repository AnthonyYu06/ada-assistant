"""Tests for ada.tools.display: show_on_screen and show_chart cards."""

from __future__ import annotations

import importlib

import pytest

from ada.tools import registry
from ada.tools.registry import ToolError, ToolResult

display = importlib.import_module("ada.tools.display")


# -- show_on_screen -----------------------------------------------------------

def test_show_on_screen_returns_markdown_card() -> None:
    result = display.show_on_screen("Packing list", "- socks\n- passport")
    assert isinstance(result, ToolResult)
    assert result.speech == "It's on your screen."
    assert result.display == {
        "kind": "markdown",
        "title": "Packing list",
        "body": "- socks\n- passport",
    }


def test_show_on_screen_strips_whitespace() -> None:
    result = display.show_on_screen("  Title  ", "  body  ")
    assert result.display["title"] == "Title"
    assert result.display["body"] == "body"


def test_show_on_screen_caps_huge_bodies() -> None:
    result = display.show_on_screen("Big", "x" * 9000)
    assert len(result.display["body"]) <= 8001
    assert result.display["body"].endswith("…")


def test_show_on_screen_rejects_empty_body() -> None:
    with pytest.raises(ToolError, match="what to show"):
        display.show_on_screen("Title", "   ")


# -- show_chart ---------------------------------------------------------------

def test_show_chart_bar_with_unit() -> None:
    result = display.show_chart(
        "Weekly temps", "bar", ["Mon", "Tue"], [70, 75.5], unit="°F"
    )
    assert isinstance(result, ToolResult)
    assert result.speech == "The chart is on your screen."
    assert result.display == {
        "kind": "chart",
        "title": "Weekly temps",
        "chart": "bar",
        "labels": ["Mon", "Tue"],
        "values": [70.0, 75.5],
        "unit": "°F",
    }


def test_show_chart_line_without_unit() -> None:
    result = display.show_chart("Trend", "line", ["a", "b", "c"], [1, 2, 3])
    assert result.display["chart"] == "line"
    assert "unit" not in result.display


def test_show_chart_rejects_bad_chart_kind() -> None:
    with pytest.raises(ToolError, match="'bar' or 'line'"):
        display.show_chart("T", "pie", ["a"], [1])


def test_show_chart_rejects_mismatched_arrays() -> None:
    with pytest.raises(ToolError, match="one value per label"):
        display.show_chart("T", "bar", ["a", "b"], [1])


def test_show_chart_rejects_empty_data() -> None:
    with pytest.raises(ToolError, match="at least one"):
        display.show_chart("T", "bar", [], [])


def test_show_chart_rejects_non_numeric_values() -> None:
    with pytest.raises(ToolError, match="numbers"):
        display.show_chart("T", "bar", ["a"], ["not-a-number"])
    with pytest.raises(ToolError, match="numbers"):
        display.show_chart("T", "bar", ["a"], [True])


def test_show_chart_rejects_too_many_points() -> None:
    n = 60
    with pytest.raises(ToolError, match="too many"):
        display.show_chart("T", "bar", ["l"] * n, [1] * n)


# -- registration -------------------------------------------------------------

def test_display_tools_registered_safe() -> None:
    for name in ("show_on_screen", "show_chart"):
        spec = registry.get(name)
        assert spec is not None
        assert spec.risk == "safe"
        assert spec.category == "display"
