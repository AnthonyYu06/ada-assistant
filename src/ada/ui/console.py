"""Console UI for Ada.

A thin, thread-safe wrapper around a rich Console. It prints:
  - one compact, dim status line per state change (via a state listener),
  - a startup banner,
  - styled user / assistant / tool / error lines.

Deliberately avoids rich.Live — a live display fights with the logging
StreamHandler that also writes to stderr.
"""

from __future__ import annotations

import threading

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..state import STATUS_ICONS, AssistantState, Status


class ConsoleUI:
    """Prints assistant activity to the terminal. All methods are thread-safe
    (rich's Console serializes writes internally; the status de-dup state is
    guarded by a lock)."""

    def __init__(self, state: AssistantState, assistant_name: str) -> None:
        self._state = state
        self._name = assistant_name
        self._console = Console(highlight=False)
        self._lock = threading.Lock()
        self._last_status: tuple[Status, str] | None = None

    # -- status stream ----------------------------------------------------
    def attach(self) -> None:
        """Register a state listener that echoes each status change."""
        self._state.add_listener(self._on_status)

    def _on_status(self, status: Status, detail: str) -> None:
        with self._lock:
            if (status, detail) == self._last_status:
                return
            self._last_status = (status, detail)
        icon = STATUS_ICONS.get(status, "•")
        line = f"{icon}  {status.value.capitalize()}…"
        if detail:
            line += f" ({detail})"
        self._console.print(Text(line, style="dim"))

    # -- startup banner -----------------------------------------------------
    def print_banner(self, info: dict[str, str]) -> None:
        """Small startup panel: assistant name on top, key/value rows inside
        (wake word, STT/TTS providers, model, mute hint, ...)."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold dim", justify="right", no_wrap=True)
        table.add_column()
        for key, value in info.items():
            table.add_row(key, Text(str(value)))
        panel = Panel(
            table,
            title=Text(self._name, style="bold cyan"),
            subtitle=Text("Ctrl+C to quit", style="dim"),
            border_style="dim",
            expand=False,
            padding=(0, 2),
        )
        self._console.print(panel)

    # -- conversation lines ---------------------------------------------------
    def print_user(self, text: str) -> None:
        line = Text("❯ you: ", style="bold cyan")
        line.append(text, style="cyan")
        self._console.print(line)

    def print_assistant(self, text: str) -> None:
        line = Text(f"● {self._name.lower()}: ", style="bold green")
        line.append(text)
        self._console.print(line)

    def print_tool(self, name: str, ok: bool) -> None:
        mark = "✓" if ok else "✗"
        self._console.print(Text(f"  ⚙ {name} {mark}", style="dim"))

    def print_error(self, text: str) -> None:
        line = Text("✗ ", style="bold red")
        line.append(text, style="red")
        self._console.print(line)
