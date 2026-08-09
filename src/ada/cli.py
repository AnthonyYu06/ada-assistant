"""Command-line interface for Ada.

Subcommands:
    ada setup     create config.yaml and download the models
    ada doctor    check dependencies, devices, models, and keys
    ada run       run the voice assistant (menu bar by default)
    ada text      chat by typing (no microphone needed)
    ada say       speak one line of text with the configured voice
    ada listen    record one utterance and print the transcript

Heavy imports happen inside the subcommand functions so `ada --help`
stays instant.
"""

from __future__ import annotations

import argparse
import sys

_WAKE_VERSION_SUFFIX = r"_v[\d.]+$"


def main(argv: list[str] | None = None) -> None:
    """Entry point (pyproject: ada = "ada.cli:main")."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        code = args.func(args)
    except KeyboardInterrupt:
        print(file=sys.stderr)
        sys.exit(130)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        _print_error(str(exc))
        sys.exit(1)
    sys.exit(int(code or 0))


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="path to config.yaml (default: auto-discover)",
    )

    parser = argparse.ArgumentParser(
        prog="ada",
        description="Ada — a privacy-conscious voice assistant for macOS.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "setup", parents=[common], help="create config.yaml and download the models"
    )
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser(
        "doctor",
        parents=[common],
        help="check dependencies, audio devices, models, and API keys",
    )
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("run", parents=[common], help="run the voice assistant")
    p.add_argument(
        "--no-gui",
        action="store_true",
        help="run in the terminal, without the on-screen HUD window",
    )
    p.add_argument(
        "--menubar",
        action="store_true",
        help="use the macOS menu-bar icon instead of the HUD window",
    )
    p.add_argument(
        "--no-menubar",
        action="store_true",
        help=argparse.SUPPRESS,  # back-compat alias for --no-gui
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser(
        "text",
        parents=[common],
        help="chat with the assistant by typing (no microphone)",
    )
    p.add_argument("--speak", action="store_true", help="also speak replies aloud")
    p.set_defaults(func=cmd_text)

    p = sub.add_parser(
        "say", parents=[common], help="speak one line of text with the configured voice"
    )
    p.add_argument("text", help="the text to speak")
    p.set_defaults(func=cmd_say)

    p = sub.add_parser(
        "listen",
        parents=[common],
        help="record one utterance from the mic and print the transcript",
    )
    p.set_defaults(func=cmd_listen)

    return parser


def _print_error(message: str) -> None:
    try:
        from rich.console import Console

        Console(stderr=True, highlight=False).print(f"[bold red]Error:[/] {message}")
    except ImportError:
        print(f"Error: {message}", file=sys.stderr)


def _load(args: argparse.Namespace, *, console_logging: bool = True):
    """Load the config and set up logging. Returns the Config."""
    from .config import load_config
    from .log import setup_logging

    cfg = load_config(args.config)
    setup_logging(cfg, console=console_logging)
    return cfg


# -- setup -----------------------------------------------------------------
def cmd_setup(args: argparse.Namespace) -> int:
    from pathlib import Path

    from rich.console import Console

    from .config import default_config_text, load_config
    from .log import setup_logging

    console = Console(highlight=False)

    target = Path(args.config).expanduser() if args.config else Path.cwd() / "config.yaml"
    if target.is_file():
        console.print(f"✅ Config already exists: [bold]{target}[/]")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(default_config_text())
        console.print(f"✅ Created config: [bold]{target}[/] — edit it to taste.")

    cfg = load_config(str(target))
    setup_logging(cfg, console=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"✅ Log directory: [bold]{cfg.log_dir}[/]")

    if cfg.wake.enabled:
        with console.status("Downloading the wake-word models (openWakeWord)…"):
            from .audio.wakeword import WakeWordDetector

            WakeWordDetector.ensure_models(cfg.wake.model)
            # Instantiate once so the model actually loads (and fails loudly
            # here rather than at run time).
            WakeWordDetector(cfg.wake.model, cfg.wake.threshold, cfg.wake.cooldown_s)
        console.print(f"✅ Wake-word model ready: [bold]{cfg.wake.model}[/]")
    else:
        console.print(
            "•  Hands-free wake word is off — Ada is triggered by the "
            f"push-to-talk hotkey ([bold]{cfg.hotkey.combo}[/]). "
            "Enable `wake` in the config to go hands-free."
        )

    # Whisper is only needed for on-device speech-to-text.
    if cfg.stt.provider == "local":
        console.print(
            f"Downloading and warming the Whisper model "
            f"([bold]{cfg.stt.local.model}[/]) — the first run can take a few minutes…"
        )
        with console.status("Loading Whisper…"):
            from .stt.local_whisper import LocalWhisperSTT

            LocalWhisperSTT(
                model_name=cfg.stt.local.model, compute_type=cfg.stt.local.compute_type
            ).warmup()
        console.print(f"✅ Whisper model ready: [bold]{cfg.stt.local.model}[/]")
    else:
        console.print(
            f"•  Speech-to-text uses the cloud provider "
            f"[bold]{cfg.stt.provider}[/] — no local Whisper model needed."
        )

    # Brain: local Ollama, or the cloud Anthropic API.
    if cfg.llm.provider == "ollama":
        oc = cfg.llm.ollama
        host = oc.host.rstrip("/")
        try:
            import httpx

            r = httpx.get(f"{host}/api/tags", timeout=4.0)
            names = [m.get("name", "") for m in r.json().get("models", [])] if r.status_code == 200 else []
            base = oc.model.split(":")[0]
            if any(n == oc.model or n.split(":")[0] == base for n in names):
                console.print(f"✅ Ollama ready: model [bold]{oc.model}[/] is pulled at {host}.")
            elif r.status_code == 200:
                console.print(
                    f"⚠️  Ollama is running but [bold]{oc.model}[/] isn't pulled yet. "
                    f"Run: [bold]ollama pull {oc.model}[/]"
                )
            else:
                console.print(f"⚠️  Ollama responded HTTP {r.status_code} at {host}.")
        except Exception:  # noqa: BLE001
            console.print(
                f"⚠️  Ollama isn't reachable at {host}. Install it from "
                f"[bold]ollama.com[/], then run [bold]ollama serve[/] and "
                f"[bold]ollama pull {oc.model}[/]."
            )
    elif cfg.anthropic_api_key:
        console.print("✅ ANTHROPIC_API_KEY is set.")
    else:
        console.print(
            "⚠️  ANTHROPIC_API_KEY is not set — add it to your environment or a "
            f".env file next to {target.name} before running the assistant."
        )

    # Voice: download the Kokoro neural model if that's the chosen TTS.
    if cfg.tts.provider == "kokoro":
        try:
            import kokoro_onnx  # noqa: F401

            from .tts.kokoro import ensure_models, models_present

            if models_present():
                console.print(f"✅ Kokoro voice ready: [bold]{cfg.tts.kokoro.voice}[/].")
            else:
                with console.status("Downloading the Kokoro voice model (~340 MB)…"):
                    ensure_models()
                console.print(f"✅ Kokoro voice ready: [bold]{cfg.tts.kokoro.voice}[/].")
        except ImportError:
            console.print(
                "⚠️  tts.provider is 'kokoro' but kokoro-onnx isn't installed — "
                "run [bold]pip install -e \".[voice]\"[/] (Ada will use the macOS "
                "voice until then)."
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"⚠️  Could not prepare the Kokoro voice: {exc}")

    console.print(
        "\nAll set. Try [bold]ada run[/] — or [bold]ada doctor[/] to double-check."
    )
    return 0


# -- doctor -----------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    import importlib
    import os
    import platform
    import re
    from pathlib import Path

    from rich.console import Console

    console = Console(highlight=False)
    cfg = _load(args, console_logging=False)

    rows: list[tuple[str, str, str]] = []
    failed = False

    def add(mark: str, check: str, detail: str) -> None:
        nonlocal failed
        if mark == "❌":
            failed = True
        rows.append((mark, check, detail))

    # Python version.
    py = platform.python_version()
    if sys.version_info >= (3, 11):
        add("✅", "Python", py)
    else:
        add("❌", "Python", f"{py} (3.11 or newer required)")

    # Config file in use.
    if cfg.config_path is not None:
        add("✅", "Config file", str(cfg.config_path))
    else:
        add("⚠️", "Config file", "none found — using built-in defaults (run `ada setup`)")

    # Core and optional imports.
    core = ("sounddevice", "openwakeword", "pysilero_vad", "faster_whisper",
            "anthropic", "yaml", "httpx")
    optional = {
        "rumps": "menu-bar icon disabled (pip install rumps)",
        "pynput": "push-to-talk hotkey disabled (pip install pynput)",
        "ddgs": "web search disabled (pip install ddgs)",
    }
    for mod in core:
        try:
            importlib.import_module(mod)
            add("✅", f"import {mod}", "ok")
        except Exception as exc:  # noqa: BLE001 - report anything (incl. native load errors)
            add("❌", f"import {mod}", str(exc))
    for mod, hint in optional.items():
        try:
            importlib.import_module(mod)
            add("✅", f"import {mod} (optional)", "ok")
        except Exception:  # noqa: BLE001
            add("⚠️", f"import {mod} (optional)", hint)

    # Input devices.
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        try:
            default_in = sd.default.device[0]
        except Exception:  # noqa: BLE001
            default_in = -1
        inputs = [
            (i, d) for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0
        ]
        if inputs:
            listed = "; ".join(
                f"[{i}] {d['name']}" + (" ← default" if i == default_in else "")
                for i, d in inputs
            )
            add("✅", "Input devices", listed)
        else:
            add("❌", "Input devices", "no audio input devices found")
    except Exception as exc:  # noqa: BLE001
        add("❌", "Input devices", str(exc))

    # Record half a second from the mic and report the peak level.
    try:
        import numpy as np
        import sounddevice as sd

        samples = int(0.5 * cfg.audio.sample_rate)
        recording = sd.rec(
            samples,
            samplerate=cfg.audio.sample_rate,
            channels=1,
            dtype="int16",
            device=cfg.audio.input_device,
        )
        sd.wait()
        peak = float(np.abs(recording).max()) / 32768.0
        if peak < 0.001:
            add(
                "⚠️",
                "Microphone",
                f"recorded 0.5 s but it was silent (peak {peak:.4f}) — "
                "check the input device and that the mic isn't hardware-muted",
            )
        else:
            add("✅", "Microphone", f"recorded 0.5 s, peak level {peak:.2f}")
    except Exception as exc:  # noqa: BLE001
        add(
            "❌",
            "Microphone",
            f"could not record ({exc}) — grant your terminal microphone access "
            "under System Settings → Privacy & Security → Microphone",
        )

    # Wake-word model files (openWakeWord downloads into its package dir).
    try:
        import openwakeword

        models_dir = Path(openwakeword.__file__).resolve().parent / "resources" / "models"
        base = re.sub(_WAKE_VERSION_SUFFIX, "", cfg.wake.model)
        found = sorted(models_dir.glob(f"{base}*.onnx")) if models_dir.is_dir() else []
        if found:
            add("✅", "Wake-word model", found[0].name)
        else:
            add("⚠️", "Wake-word model", f"{cfg.wake.model} not downloaded — run `ada setup`")
    except Exception as exc:  # noqa: BLE001
        add("⚠️", "Wake-word model", f"could not check: {exc}")

    # Whisper model cache (best effort: look for the HF hub snapshot).
    if cfg.stt.provider == "local":
        hub = Path.home() / ".cache" / "huggingface" / "hub"
        matches = (
            sorted(hub.glob(f"models--*whisper*{cfg.stt.local.model}*"))
            if hub.is_dir()
            else []
        )
        if matches:
            add("✅", "Whisper model cache", matches[0].name)
        else:
            add(
                "⚠️",
                "Whisper model cache",
                f"no cached '{cfg.stt.local.model}' model found — "
                "run `ada setup` (or it downloads on first use)",
            )
    else:
        add("✅", "STT provider", f"cloud ({cfg.stt.provider}) — no local model needed")

    # Brain / LLM provider.
    if cfg.llm.provider == "ollama":
        oc = cfg.llm.ollama
        host = oc.host.rstrip("/")
        try:
            import httpx

            r = httpx.get(f"{host}/api/tags", timeout=4.0)
            if r.status_code == 200:
                names = [m.get("name", "") for m in r.json().get("models", [])]
                base = oc.model.split(":")[0]
                present = any(n == oc.model or n.split(":")[0] == base for n in names)
                if present:
                    add("✅", "Ollama (local brain)", f"{host}; model '{oc.model}' pulled")
                else:
                    add(
                        "❌",
                        "Ollama (local brain)",
                        f"reachable, but '{oc.model}' not pulled — run: ollama pull {oc.model}",
                    )
            else:
                add("❌", "Ollama (local brain)", f"HTTP {r.status_code} at {host}")
        except Exception as exc:  # noqa: BLE001
            add(
                "❌",
                "Ollama (local brain)",
                f"not reachable at {host} ({exc}) — install from ollama.com, then `ollama serve`",
            )
    else:
        add("✅", "LLM provider", f"anthropic ({cfg.llm.anthropic.model})")

    # API keys (masked). The Anthropic key is only required with the cloud brain.
    def _mask(value: str) -> str:
        return value[:5] + "…"

    if cfg.anthropic_api_key:
        add("✅", "ANTHROPIC_API_KEY", _mask(cfg.anthropic_api_key))
    elif cfg.llm.provider == "anthropic":
        add("❌", "ANTHROPIC_API_KEY", "absent — required for the anthropic provider (environment or .env)")
    else:
        add("⚠️", "ANTHROPIC_API_KEY", "absent (only needed if you switch llm.provider to anthropic)")
    if cfg.deepgram_api_key:
        add("✅", "DEEPGRAM_API_KEY", _mask(cfg.deepgram_api_key))
    else:
        add("⚠️", "DEEPGRAM_API_KEY", "absent (optional — cloud speech-to-text)")
    if cfg.elevenlabs_api_key:
        add("✅", "ELEVENLABS_API_KEY", _mask(cfg.elevenlabs_api_key))
    else:
        add("⚠️", "ELEVENLABS_API_KEY", "absent (optional — natural cloud voice)")

    # Voice (TTS).
    if cfg.tts.provider == "kokoro":
        try:
            import kokoro_onnx  # noqa: F401

            from .tts.kokoro import models_present

            if models_present():
                add("✅", "Voice (kokoro)", f"local neural voice '{cfg.tts.kokoro.voice}' ready")
            else:
                add(
                    "❌",
                    "Voice (kokoro)",
                    "model not downloaded — run `ada setup` (falls back to macOS 'say')",
                )
        except ImportError:
            add(
                "❌",
                "Voice (kokoro)",
                "kokoro-onnx not installed — pip install -e \".[voice]\" (falls back to 'say')",
            )
    else:
        add("✅", "Voice (TTS)", f"{cfg.tts.provider} ({cfg.tts.say.voice if cfg.tts.provider == 'say' else '—'})")

    # macOS binaries.
    for path, purpose in (
        ("/usr/bin/say", "local text-to-speech"),
        ("/usr/bin/osascript", "AppleScript tools and confirmation dialogs"),
    ):
        if os.path.isfile(path):
            add("✅", path, "present")
        else:
            add("❌", path, f"missing — {purpose} unavailable (not macOS?)")

    from rich.table import Table

    table = Table(show_header=True, header_style="bold", pad_edge=False)
    table.add_column("", width=2)
    table.add_column("Check", no_wrap=True)
    table.add_column("Details", overflow="fold")
    for mark, check, detail in rows:
        table.add_row(mark, check, detail)
    console.print(table)

    if failed:
        console.print("[bold red]Some checks failed.[/] Fix the ❌ items above and re-run.")
        return 1
    console.print("[bold green]All good.[/]")
    return 0


# -- run ----------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    import logging
    import threading

    cfg = _load(args)

    from .main import Assistant
    from .state import AssistantState
    from .ui import (
        ConsoleUI,
        HotkeyListener,
        hotkey_available,
        menubar_available,
        run_menubar,
    )

    log = logging.getLogger("ada.cli")

    state = AssistantState()
    console = ConsoleUI(state, cfg.assistant.name)
    console.attach()

    assistant = Assistant(cfg, state, console)
    assistant.build()  # RuntimeError (mic / API key) is handled in main()

    banner: dict[str, str] = {
        "wake word": cfg.wake.model if cfg.wake.enabled else "disabled",
        "stt": (
            f"{cfg.stt.provider} ({cfg.stt.local.model})"
            if cfg.stt.provider == "local"
            else cfg.stt.provider
        ),
        "tts": cfg.tts.provider,
        "brain": f"{cfg.llm.provider} ({cfg.llm.active_model})",
        "config": str(cfg.config_path) if cfg.config_path else "built-in defaults",
        "logs": str(cfg.log_dir),
        "controls": 'tap the orb / press the hotkey to talk · type in the box · mute & quit in the window',
    }
    if cfg.hotkey.enabled:
        banner["hotkey"] = cfg.hotkey.combo
    console.print_banner(banner)

    # Warm the models on a background thread while the UI comes up.
    threading.Thread(target=assistant.warmup, name="ada-warmup", daemon=True).start()

    listener = None
    if cfg.hotkey.enabled:
        if hotkey_available():
            try:
                listener = HotkeyListener(cfg.hotkey.combo, assistant.trigger)
                listener.start()
            except (RuntimeError, ValueError) as exc:
                listener = None
                console.print_error(f"Hotkey disabled: {exc}")
        else:
            console.print_error(
                "hotkey.enabled is true but pynput is not installed — hotkey disabled."
            )

    def _worker() -> None:
        try:
            assistant.run()
        except Exception as exc:  # noqa: BLE001 - surface, then stop the app
            log.exception("Assistant loop crashed")
            console.print_error(str(exc))
            state.request_shutdown()

    def _install_signals() -> None:
        import signal

        def _stop(_signum: int, _frame: object) -> None:
            state.request_shutdown()

        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)

    def _run_console() -> None:
        _install_signals()
        assistant.run()  # blocks the main thread until shutdown

    def _run_menubar() -> None:
        worker = threading.Thread(target=_worker, name="ada-loop", daemon=True)
        run_menubar(state, cfg.assistant.name, worker)  # blocks the main thread

    # Choose the UI surface: HUD window (default), menu bar, or plain terminal.
    want_console = args.no_gui or args.no_menubar
    want_menubar = args.menubar and not want_console
    try:
        if want_console:
            _run_console()
        elif want_menubar:
            if menubar_available():
                _run_menubar()
            else:
                console.print_error(
                    "The menu bar needs rumps (pip install -e \".[menubar]\") — "
                    "running in the terminal instead."
                )
                _run_console()
        else:
            from .ui.hud import hud_available, run_hud

            if cfg.ui.hud and hud_available():
                _install_signals()
                worker = threading.Thread(target=_worker, name="ada-loop", daemon=True)
                worker.start()
                run_hud(state, assistant, cfg)  # blocks the main thread
            elif cfg.ui.menubar and menubar_available():
                log.info("HUD unavailable — using the menu bar. Install the HUD with '.[gui]'.")
                _run_menubar()
            else:
                console.print_error(
                    "The on-screen HUD needs pywebview (pip install -e \".[gui]\"). "
                    "Running in the terminal — use --no-gui to silence this."
                )
                _run_console()
    finally:
        state.request_shutdown()
        if listener is not None:
            listener.stop()
        assistant.shutdown()
    return 0


# -- text -----------------------------------------------------------------------
def cmd_text(args: argparse.Namespace) -> int:
    cfg = _load(args)

    from .main import run_text_repl

    run_text_repl(cfg, speak=args.speak)
    return 0


# -- say ------------------------------------------------------------------------
def cmd_say(args: argparse.Namespace) -> int:
    cfg = _load(args)

    from .tts import create_tts

    create_tts(cfg).speak(args.text)
    return 0


# -- listen -----------------------------------------------------------------------
def cmd_listen(args: argparse.Namespace) -> int:
    cfg = _load(args)

    from rich.console import Console

    from .audio import MicStream, UtteranceRecorder
    from .stt import create_stt

    console = Console(highlight=False)

    stt = create_stt(cfg)
    with console.status(f"Loading the {cfg.stt.provider} STT model…"):
        stt.warmup()

    mic = MicStream(cfg)
    recorder = UtteranceRecorder(cfg)
    mic.start()
    try:
        console.print("[bold]Listening… speak now.[/bold]")
        utterance = recorder.record(mic)
    finally:
        mic.stop()

    if utterance is None:
        console.print("[yellow]No speech detected.[/yellow]")
        return 1

    result = stt.transcribe(utterance, cfg.audio.sample_rate)
    text = result.text.strip()
    if text:
        console.print(f"[bold green]Transcript:[/] {text}")
    else:
        console.print("[yellow]Recorded audio, but the transcript came back empty.[/yellow]")
    console.print(
        f"[dim]{result.audio_s:.1f}s of audio, transcribed in "
        f"{result.latency_s:.2f}s via {result.provider}.[/dim]"
    )
    return 0


if __name__ == "__main__":
    main()
