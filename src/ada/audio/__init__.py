"""Audio subsystem: mic capture, wake word, VAD endpointing, and chimes.

Submodules are imported lazily (PEP 562) so that merely importing this
package does not pull in ``sounddevice`` / ``pysilero_vad``. Text mode
(``ada text``, ``build(voice=False)``) can therefore run even when the
audio stack is broken; only the voice paths that actually touch these
names trigger the heavy imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .capture import MicStream
    from .chime import chime_done, chime_error, chime_timer, chime_wake
    from .vad import UtteranceRecorder
    from .wakeword import WakeWordDetector

__all__ = [
    "MicStream",
    "UtteranceRecorder",
    "WakeWordDetector",
    "chime_wake",
    "chime_done",
    "chime_error",
    "chime_timer",
]

# Exported name -> submodule that defines it.
_SUBMODULES: dict[str, str] = {
    "MicStream": ".capture",
    "UtteranceRecorder": ".vad",
    "WakeWordDetector": ".wakeword",
    "chime_wake": ".chime",
    "chime_done": ".chime",
    "chime_error": ".chime",
    "chime_timer": ".chime",
}


def __getattr__(name: str) -> Any:
    """Lazily import an exported name from its submodule (PEP 562)."""
    module_name = _SUBMODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache so future lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
