"""Safety layer contracts shared by the policy engine, confirmers, and main loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..tools.registry import ToolSpec


@dataclass
class Decision:
    """Result of evaluating a tool call against the permission policy."""

    allowed: bool
    needs_confirmation: bool = False
    reason: str = ""


class PermissionPolicy(Protocol):
    def evaluate(self, spec: ToolSpec, args: dict[str, Any]) -> Decision:
        """Decide whether this tool call may run, and whether to confirm first."""
        ...


class Confirmer(Protocol):
    def confirm(self, title: str, message: str) -> bool:
        """Ask the user to approve an action. Blocking. True = approved."""
        ...
