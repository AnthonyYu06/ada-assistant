"""Permission policy: decides whether a tool call may run.

The policy is deliberately conservative: any internal error while
evaluating a call fails CLOSED (the call is denied).
"""

from __future__ import annotations

import logging
import urllib.parse
from pathlib import Path
from typing import Any

from ..config import Config
from ..tools.registry import ToolSpec
from .base import Decision

log = logging.getLogger("ada.safety")


class Policy:
    """Implements the PermissionPolicy protocol against the loaded Config."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    def evaluate(self, spec: ToolSpec, args: dict[str, Any]) -> Decision:
        """Decide whether this tool call may run, and whether to confirm first.

        Never raises: on any internal error the call is denied (fail closed).
        """
        try:
            return self._evaluate(spec, args)
        except Exception:  # noqa: BLE001 - fail closed, never propagate
            log.exception("Policy evaluation failed for tool %r", spec.name)
            return Decision(allowed=False, reason="policy error")

    # -- internals --------------------------------------------------------

    def _evaluate(self, spec: ToolSpec, args: dict[str, Any]) -> Decision:
        perms = self._cfg.permissions
        name = spec.name

        # Tool-specific checks.
        if name in ("open_app", "quit_app"):
            denial = self._check_app(args)
            if denial is not None:
                return denial
        elif name in ("open_url", "fetch_url"):
            denial = self._check_url(args)
            if denial is not None:
                return denial
        elif name == "open_path":
            denial = self._check_path(args)
            if denial is not None:
                return denial
        elif name == "set_timer":
            denial = self._check_timer(args)
            if denial is not None:
                return denial

        # Confirmation requirement.
        needs_confirmation = spec.risk == "confirm" or name in perms.always_confirm
        return Decision(allowed=True, needs_confirmation=needs_confirmation)

    # Suffixes that make open_path launch code rather than open a document.
    _LAUNCHABLE_SUFFIXES = frozenset(
        {".app", ".command", ".terminal", ".workflow", ".scpt", ".applescript", ".sh", ".tool"}
    )

    def _check_app(self, args: dict[str, Any]) -> Decision | None:
        app = str(args.get("app_name") or "")
        blocked = {b.casefold() for b in self._cfg.permissions.blocked_apps}
        # Canonicalize so a bundle path or "Name.app" suffix can't bypass the
        # name-only blocked list.
        normalized = app.rstrip("/")
        if "/" in normalized:
            normalized = normalized.rsplit("/", 1)[-1]
        if normalized.casefold().endswith(".app"):
            normalized = normalized[: -len(".app")]
        if {app.casefold(), normalized.casefold()} & blocked:
            return Decision(allowed=False, reason=f"{app} is on the blocked apps list")
        return None

    def _check_url(self, args: dict[str, Any]) -> Decision | None:
        url = str(args.get("url") or "")
        scheme = urllib.parse.urlsplit(url).scheme.lower()
        allowed_schemes = {s.lower() for s in self._cfg.permissions.url_schemes}
        if scheme not in allowed_schemes:
            shown = scheme if scheme else "(none)"
            return Decision(
                allowed=False,
                reason=f"URL scheme '{shown}' is not allowed.",
            )
        return None

    def _check_path(self, args: dict[str, Any]) -> Decision | None:
        raw = str(args.get("path") or "")
        if not raw:
            return Decision(allowed=False, reason="No path was provided.")
        path = Path(raw).expanduser().resolve()
        roots: list[Path] = list(self._cfg.permissions.file_roots_resolved())
        roots += [
            Path(d).expanduser().resolve()
            for d in self._cfg.permissions.allowed_app_dirs
        ]
        if not any(path.is_relative_to(root) for root in roots):
            return Decision(
                allowed=False,
                reason="That path is outside your allowed folders.",
            )
        # Membership in allowed_app_dirs (e.g. /Applications) is NOT auto-safe:
        # opening an .app bundle or a script executes code. Require confirmation
        # for launchable paths, and honor the blocked-apps list for .app bundles.
        suffix = path.suffix.lower()
        if suffix in self._LAUNCHABLE_SUFFIXES:
            if suffix == ".app":
                blocked = {b.casefold() for b in self._cfg.permissions.blocked_apps}
                bundle_name = path.name[: -len(".app")] if path.name.lower().endswith(".app") else path.name
                if {bundle_name.casefold(), path.name.casefold()} & blocked:
                    return Decision(
                        allowed=False,
                        reason=f"{path.name} is on the blocked apps list",
                    )
            return Decision(
                allowed=True,
                needs_confirmation=True,
                reason=f"Opening {path.name} can run code.",
            )
        return None

    def _check_timer(self, args: dict[str, Any]) -> Decision | None:
        raw = args.get("duration_seconds")
        try:
            seconds = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return Decision(allowed=False, reason="Invalid timer duration.")
        if not 1 <= seconds <= 86400:
            return Decision(
                allowed=False,
                reason="Timer duration must be between 1 second and 24 hours.",
            )
        return None
