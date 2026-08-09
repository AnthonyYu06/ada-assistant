"""Tests for ada.safety.permissions.Policy: deny rules, confirmation, fail-closed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ada.config import Config
from ada.safety.base import Decision
from ada.safety.permissions import Policy
from ada.tools.registry import ToolSpec


def _spec(name: str, risk: str = "safe") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="test tool",
        input_schema={"type": "object"},
        handler=lambda **kw: "ok",
        risk=risk,  # type: ignore[arg-type]
    )


def _policy(**permission_overrides: Any) -> Policy:
    cfg = Config()
    for key, value in permission_overrides.items():
        setattr(cfg.permissions, key, value)
    return Policy(cfg)


class _ExplodingStr:
    """An arg value whose str() raises, to exercise the fail-closed path."""

    def __str__(self) -> str:
        raise RuntimeError("boom")


# -- blocked apps -----------------------------------------------------------

def test_blocked_app_denied_case_insensitive() -> None:
    policy = _policy(blocked_apps=["Terminal"])
    for variant in ("Terminal", "terminal", "TERMINAL", "tErMiNaL"):
        decision = policy.evaluate(_spec("open_app"), {"app_name": variant})
        assert decision.allowed is False
        assert "blocked" in decision.reason.lower()


def test_blocked_app_applies_to_quit_app() -> None:
    policy = _policy(blocked_apps=["Terminal"])
    decision = policy.evaluate(_spec("quit_app"), {"app_name": "terminal"})
    assert decision.allowed is False


def test_blocked_app_denied_via_dot_app_suffix_and_path() -> None:
    # A name-only blocked entry must not be bypassable via a ".app" suffix or
    # a full bundle path.
    policy = _policy(blocked_apps=["Blocked"])
    for variant in (
        "Blocked.app",
        "blocked.app",
        "/Applications/Blocked.app",
        "/Applications/Blocked.app/",
    ):
        decision = policy.evaluate(_spec("open_app"), {"app_name": variant})
        assert decision.allowed is False, variant
        assert "blocked" in decision.reason.lower()


def test_unblocked_app_allowed() -> None:
    policy = _policy(blocked_apps=["Terminal"])
    decision = policy.evaluate(_spec("open_app"), {"app_name": "Safari"})
    assert decision.allowed is True


# -- URL schemes ------------------------------------------------------------

def test_disallowed_url_scheme_denied() -> None:
    policy = _policy()
    for url in (
        "file:///etc/passwd",
        "javascript:alert(1)",
        "ftp://example.com/x",
        "not-a-url",  # no scheme at all
        "",
    ):
        decision = policy.evaluate(_spec("open_url"), {"url": url})
        assert decision.allowed is False, url
        assert "scheme" in decision.reason.lower()


def test_allowed_url_schemes() -> None:
    policy = _policy()
    for url in ("https://example.com", "http://example.com/page"):
        decision = policy.evaluate(_spec("open_url"), {"url": url})
        assert decision.allowed is True, url


def test_url_check_applies_to_fetch_url() -> None:
    policy = _policy()
    assert policy.evaluate(_spec("fetch_url"), {"url": "file:///x"}).allowed is False
    assert policy.evaluate(_spec("fetch_url"), {"url": "https://a.com"}).allowed is True


def test_url_scheme_list_is_configurable() -> None:
    policy = _policy(url_schemes=["https"])
    assert policy.evaluate(_spec("open_url"), {"url": "http://a.com"}).allowed is False
    assert policy.evaluate(_spec("open_url"), {"url": "https://a.com"}).allowed is True


# -- open_path file roots ---------------------------------------------------

def test_open_path_inside_file_roots_allowed(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    policy = _policy(file_roots=[str(docs)], allowed_app_dirs=[])
    decision = policy.evaluate(
        _spec("open_path"), {"path": str(docs / "sub" / "report.pdf")}
    )
    assert decision.allowed is True


def test_open_path_outside_file_roots_denied(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    policy = _policy(file_roots=[str(docs)], allowed_app_dirs=[])
    decision = policy.evaluate(
        _spec("open_path"), {"path": str(tmp_path / "elsewhere" / "secret.txt")}
    )
    assert decision.allowed is False
    assert decision.reason


def test_open_path_dotdot_traversal_denied(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    policy = _policy(file_roots=[str(docs)], allowed_app_dirs=[])
    sneaky = str(docs / ".." / "outside.txt")
    decision = policy.evaluate(_spec("open_path"), {"path": sneaky})
    assert decision.allowed is False


def test_open_path_empty_denied(tmp_path: Path) -> None:
    policy = _policy(file_roots=[str(tmp_path)], allowed_app_dirs=[])
    assert policy.evaluate(_spec("open_path"), {"path": ""}).allowed is False
    assert policy.evaluate(_spec("open_path"), {}).allowed is False


def test_open_path_allowed_app_dirs_count_as_roots(tmp_path: Path) -> None:
    apps = tmp_path / "apps"
    apps.mkdir()
    policy = _policy(file_roots=[], allowed_app_dirs=[str(apps)])
    decision = policy.evaluate(
        _spec("open_path"), {"path": str(apps / "Safari.app")}
    )
    assert decision.allowed is True


# -- confirmation -----------------------------------------------------------

def test_risk_confirm_needs_confirmation() -> None:
    policy = _policy()
    decision = policy.evaluate(_spec("close_window", risk="confirm"), {})
    assert decision.allowed is True
    assert decision.needs_confirmation is True


def test_always_confirm_needs_confirmation() -> None:
    # quit_app is risk-safe here, but on the default always_confirm list.
    policy = _policy()
    decision = policy.evaluate(_spec("quit_app"), {"app_name": "Safari"})
    assert decision.allowed is True
    assert decision.needs_confirmation is True


def test_always_confirm_is_configurable() -> None:
    policy = _policy(always_confirm=["tell_joke"])
    decision = policy.evaluate(_spec("tell_joke"), {})
    assert decision.allowed is True
    assert decision.needs_confirmation is True


def test_safe_tool_no_confirmation() -> None:
    policy = _policy()
    decision = policy.evaluate(_spec("get_time"), {})
    assert decision == Decision(allowed=True, needs_confirmation=False, reason="")


# -- set_timer bounds -------------------------------------------------------

def test_timer_bounds() -> None:
    policy = _policy()
    spec = _spec("set_timer")
    assert policy.evaluate(spec, {"duration_seconds": 60}).allowed is True
    assert policy.evaluate(spec, {"duration_seconds": 1}).allowed is True
    assert policy.evaluate(spec, {"duration_seconds": 86400}).allowed is True
    assert policy.evaluate(spec, {"duration_seconds": 0}).allowed is False
    assert policy.evaluate(spec, {"duration_seconds": 86401}).allowed is False
    assert policy.evaluate(spec, {"duration_seconds": "abc"}).allowed is False
    assert policy.evaluate(spec, {"duration_seconds": None}).allowed is False
    assert policy.evaluate(spec, {}).allowed is False


# -- fail closed ------------------------------------------------------------

def test_policy_never_raises_and_fails_closed() -> None:
    policy = _policy(blocked_apps=["Terminal"])
    # str() on the arg raises inside _check_app; evaluate must swallow it
    # and deny, never propagate.
    decision = policy.evaluate(_spec("open_app"), {"app_name": _ExplodingStr()})
    assert decision.allowed is False
    assert decision.needs_confirmation is False
    assert decision.reason == "policy error"


def test_policy_fails_closed_for_path_errors() -> None:
    policy = _policy()
    decision = policy.evaluate(_spec("open_path"), {"path": _ExplodingStr()})
    assert decision.allowed is False
