"""The assistant itself: wiring + the voice event loop.

Assistant owns every subsystem (audio capture, wake word, VAD, STT, brain,
TTS, safety) and runs the main loop:

    wake word -> record utterance -> transcribe -> local intents ->
    brain (streaming, tools) -> speak sentence by sentence

The loop is designed to run on a worker thread when the menu bar owns the
process main thread (`ada run`), or directly on the main thread
(`ada run --no-menubar`). Each conversational turn runs on its own
short-lived thread so the loop keeps scanning the mic for barge-in.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import TYPE_CHECKING

from .audio.chime import chime_done, chime_error, chime_wake
from .brain import BaseBrain, create_brain
from .config import Config
from .perf import TurnTimer
from .safety import AuditLog, Confirmer, ConsoleConfirmer, DialogConfirmer, Policy
from .state import AssistantState, Status
from .tts import Speaker, create_tts
from .ui import events as ui_events

if TYPE_CHECKING:
    from .audio import MicStream, UtteranceRecorder, WakeWordDetector
    from .stt.base import STTProvider
    from .ui.console import ConsoleUI

log = logging.getLogger("ada.main")

# Local intents recognized without the LLM (compared after normalization).
_CANCEL_PHRASES = frozenset({"stop", "cancel", "never mind", "nevermind", "forget it"})
_MUTE_PHRASES = frozenset({"stop listening", "mute", "go to sleep"})
_SHUTDOWN_PHRASES = frozenset({"shut down", "shutdown", "quit", "goodbye ada"})

# How long to wait for a reply to finish playing before giving up.
_SPEAK_TIMEOUT_S = 120.0
_ERROR_SPEAK_TIMEOUT_S = 15.0
# How long the main loop waits for a cancelled turn to wind down (barge-in).
_BARGE_IN_JOIN_S = 5.0


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for intent matching."""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return " ".join(cleaned.split())


class Assistant:
    """Owns all subsystems and runs the wake -> listen -> think -> speak loop."""

    def __init__(
        self,
        cfg: Config,
        state: AssistantState,
        console: ConsoleUI | None = None,
    ) -> None:
        self._cfg = cfg
        self._state = state
        self._console = console

        # External activation (push-to-talk hotkey) — treated like a wake word.
        self._activate = threading.Event()
        self._turn_thread: threading.Thread | None = None
        self._cancel_event: threading.Event | None = None
        # Set while the utterance recorder owns the mic queue; the run loop
        # uses this (not the status) as the source of truth for whether to
        # stop reading/processing mic frames.
        self._recording = threading.Event()
        # Speaker generation captured at the start of a turn, so sentences
        # emitted by a barged-in turn can't play during the next one.
        self._turn_gen: int | None = None
        # Timer/reminder announcements deferred while a recording is active;
        # flushed on return to IDLE so they don't play into the utterance.
        self._pending_announcements: list[str] = []
        self._announce_lock = threading.Lock()
        # Serializes turn spawning so the voice loop and typed input (HUD)
        # can't start two turns at once.
        self._turn_lock = threading.Lock()
        self._built = False
        self._voice = False

        # Wired by build():
        self.brain: BaseBrain | None = None
        self.speaker: Speaker | None = None
        self._confirmer: Confirmer | None = None
        self._audit: AuditLog | None = None
        self._stt: STTProvider | None = None
        self._mic: MicStream | None = None
        self._wake: WakeWordDetector | None = None
        self._recorder: UtteranceRecorder | None = None
        self._tts_name = "none"

    # -- wiring ------------------------------------------------------------
    def build(
        self,
        *,
        voice: bool = True,
        tts: bool = True,
        confirmer: Confirmer | None = None,
    ) -> None:
        """Construct and wire every subsystem.

        voice=False skips the microphone / wake word / VAD / STT (text mode);
        those modules are only imported when needed, so text mode works even
        if the audio dependencies are broken. tts=False skips the voice
        output entirely. `confirmer` overrides the native dialog confirmer
        (the text REPL passes a ConsoleConfirmer).
        """
        cfg = self._cfg

        from . import tools as tools_pkg
        from .tools import productivity

        tools_pkg.load_all(cfg)
        # Timer announcements speak through the Speaker; _announce is safe to
        # call even before (or without) a Speaker being built.
        productivity.set_announcer(self._announce)

        policy = Policy(cfg)
        self._confirmer = confirmer if confirmer is not None else DialogConfirmer()
        self._audit = AuditLog(cfg.log_dir / "audit.jsonl", cfg.logging.audit)
        self.brain = create_brain(cfg, policy, self._confirmer, self._audit)
        if cfg.memory.enabled:
            # Same store instance the remember/recall tools use, so facts
            # saved mid-conversation are recallable on the very next turn.
            try:
                from .tools.memory_tools import get_store

                self.brain.knowledge = get_store()
                log.info("Long-term memory active (%s)", cfg.memory_path)
            except Exception:  # noqa: BLE001 - memory is an optional extra
                log.exception("Could not initialize long-term memory")

        if tts:
            provider = create_tts(cfg)
            self._tts_name = provider.name
            self.speaker = Speaker(provider)

        if voice:
            from .audio import MicStream, UtteranceRecorder, WakeWordDetector
            from .stt import create_stt

            self._stt = create_stt(cfg)
            self._mic = MicStream(cfg)
            self._recorder = UtteranceRecorder(cfg)
            if cfg.wake.enabled:
                self._wake = WakeWordDetector(
                    cfg.wake.model, cfg.wake.threshold, cfg.wake.cooldown_s
                )
            else:
                log.info(
                    "Wake word disabled — activation is via the hotkey only."
                )

        self._voice = voice
        self._built = True
        log.info(
            "Assistant built (voice=%s, stt=%s, tts=%s, wake=%s)",
            voice,
            getattr(self._stt, "name", "none"),
            self._tts_name,
            cfg.wake.model if self._wake is not None else "disabled",
        )

    def warmup(self) -> None:
        """Load the heavy models so the first turn is fast. Sets IDLE when done."""
        if self._stt is not None:
            start = time.monotonic()
            self._stt.warmup()
            log.info("STT warmup finished in %.1fs", time.monotonic() - start)
        if self.brain is not None:
            start = time.monotonic()
            self.brain.warmup()
            log.info("Brain warmup finished in %.1fs", time.monotonic() - start)
        # The state stays STARTING for the whole warmup; only then go IDLE,
        # and only if nothing else moved it on in the meantime.
        self._state.set_status_if(Status.STARTING, Status.IDLE)

    # -- the event loop ------------------------------------------------------
    def run(self) -> None:
        """The worker loop: scan the mic for the wake word, spawn turns.

        Runs until state.request_shutdown() is called. Designed to run on a
        daemon thread when the menu bar owns the main thread, or directly on
        the main thread with --no-menubar.
        """
        if not self._built or not self._voice:
            raise RuntimeError("Assistant.run() requires build(voice=True) first")
        assert self._mic is not None and self._recorder is not None
        assert self._audit is not None

        state = self._state
        mic = self._mic
        mic.start()
        self._audit.record(
            "startup",
            stt=getattr(self._stt, "name", "none"),
            tts=self._tts_name,
            model=self._cfg.llm.active_model,
            wake=self._cfg.wake.model if self._wake is not None else "disabled",
        )
        log.info("Assistant loop running")

        prev_muted = False
        # Monotonic timestamp of the last frame observed while the speaker was
        # playing; used for the post-playback wake hangover window.
        speaking_seen_at = 0.0
        try:
            while not state.shutting_down:
                muted = state.muted
                if muted and not prev_muted:
                    # Mute engaged (menu bar / voice) while a turn may be in
                    # flight: cancel it so it releases the mic instead of
                    # recording into a muted session.
                    if self._cancel_event is not None:
                        self._cancel_event.set()
                    self._cancel_turn(join_timeout=1.0)
                prev_muted = muted

                if self._recording.is_set():
                    # The utterance recorder owns the mic queue right now.
                    time.sleep(0.05)
                    continue

                frame = mic.read(timeout=0.5)

                if muted:
                    # Frames are read and discarded; the wake detector is not
                    # fed, and any pending hotkey activation is dropped too.
                    self._activate.clear()
                    continue

                activated = self._activate.is_set()
                if activated:
                    self._activate.clear()

                # While (and shortly after) our own TTS is playing, raw mic
                # frames contain the assistant's voice — there is no acoustic
                # echo cancellation — so a soft wake-word match may just be the
                # reply echoing back. Barge-in still needs wake during
                # SPEAKING, so instead of suppressing we raise the confidence
                # bar; a clearly-spoken wake still crosses 0.7 and barges in.
                speaking = self.speaker is not None and self.speaker.is_speaking
                now = time.monotonic()
                if speaking:
                    speaking_seen_at = now
                hangover = now - speaking_seen_at < 0.8

                detected = activated
                if not detected and frame is not None and self._wake is not None:
                    detected = self._wake.process(frame)
                    if detected and (speaking or hangover):
                        min_score = max(self._cfg.wake.threshold, 0.7)
                        if self._wake.last_score < min_score:
                            detected = False
                if not detected:
                    continue

                self._audit.record(
                    "wake", source="hotkey" if activated else "wake_word"
                )
                with self._turn_lock:
                    if not self._prepare_for_turn():
                        continue

                    if activated:
                        # Hotkey activation: the mic queue is full of pre-trigger
                        # audio; clear it here on the run-loop thread. On the wake
                        # path we keep the queue so a continuously-spoken command's
                        # onset (captured after the wake word) is not clipped.
                        mic.clear()

                    cancel = threading.Event()
                    self._cancel_event = cancel
                    state.set_status(Status.LISTENING)
                    thread = threading.Thread(
                        target=self._run_turn,
                        args=(cancel, activated),
                        name="ada-turn",
                        daemon=True,
                    )
                    self._turn_thread = thread
                    thread.start()
        finally:
            self._cancel_turn(join_timeout=3.0)
            log.info("Assistant loop stopped")

    def trigger(self) -> None:
        """External activation (push-to-talk hotkey) — treated as a wake word."""
        self._activate.set()

    def shutdown(self) -> None:
        """Release everything: audit the shutdown, stop TTS, close the mic."""
        log.info("Shutting down")
        self._cancel_turn(join_timeout=3.0)
        if self._audit is not None:
            self._audit.record("shutdown")
        if self.speaker is not None:
            self.speaker.close()
        if self._mic is not None:
            self._mic.stop()
        close = getattr(self.brain, "close", None)
        if callable(close):
            close()

    # -- turn management -----------------------------------------------------
    def _prepare_for_turn(self) -> bool:
        """Make room for a new turn; handles barge-in. True = go ahead."""
        thread = self._turn_thread
        if thread is None or not thread.is_alive():
            return True
        if self._state.status not in (
            Status.THINKING,
            Status.SPEAKING,
            Status.ERROR,
        ):
            # A turn exists but is in a transition state — ignore the trigger.
            # ERROR is included so an activation lands even while a failed turn
            # is playing its error apology.
            return False
        log.info("Barge-in: cancelling the current turn")
        if self._audit is not None:
            self._audit.record("barge_in")
        cancel = self._cancel_event
        if cancel is not None:
            cancel.set()
        if self.speaker is not None:
            self.speaker.stop()
        thread.join(timeout=_BARGE_IN_JOIN_S)
        if thread.is_alive():
            log.warning("Previous turn is still winding down; ignoring activation")
            return False
        return True

    def _cancel_turn(self, join_timeout: float) -> None:
        """Cancel any in-flight turn and wait briefly for it to end."""
        cancel = self._cancel_event
        if cancel is not None:
            cancel.set()
        thread = self._turn_thread
        if thread is not None and thread.is_alive():
            if self.speaker is not None:
                self.speaker.stop()
            thread.join(timeout=join_timeout)

    def _run_turn(self, cancel: threading.Event, from_hotkey: bool = False) -> None:
        """One conversational turn: record -> transcribe -> answer -> speak.

        `from_hotkey` records whether this turn was activated by the push-to-
        talk hotkey (the mic queue was cleared on the run-loop thread) rather
        than the wake word (queue kept so the command onset isn't clipped).
        """
        cfg = self._cfg
        state = self._state
        assert self._mic is not None and self._recorder is not None
        assert self._stt is not None and self.brain is not None
        assert self._audit is not None
        try:
            state.set_status(Status.LISTENING)
            chime_wake(cfg.ui.chimes)
            timer = TurnTimer(publish=cfg.ui.show_timings)
            # Do NOT clear the mic on the wake path: frames captured just after
            # the wake word hold the onset of a continuously-spoken command.
            # Hotkey activations are cleared on the run-loop thread beforehand.
            self._recording.set()
            try:
                utterance = self._recorder.record(self._mic, cancel)
            finally:
                self._recording.clear()
            timer.mark("listen")
            if cancel.is_set():
                self._finish_turn()
                return
            if utterance is None:
                log.info("No speech captured")
                chime_error(cfg.ui.chimes)
                self._finish_turn()
                return

            state.set_status(Status.THINKING, "transcribing")
            result = self._stt.transcribe(utterance, cfg.audio.sample_rate)
            timer.mark("transcribe")
            text = result.text.strip()
            if not text:
                log.info("Empty transcript")
                chime_error(cfg.ui.chimes)
                self._finish_turn()
                return
            log.info(
                "Heard %r (%.1fs audio, transcribed in %.2fs via %s)",
                text,
                result.audio_s,
                result.latency_s,
                result.provider,
            )
            self._audit.record(
                "transcript",
                text=text,
                provider=result.provider,
                audio_s=round(result.audio_s, 2),
                latency_s=round(result.latency_s, 2),
            )
            self._answer(text, cancel, timer)
        except Exception:
            log.exception("Turn failed")
            self._speak_error()

    def submit_text(self, text: str) -> None:
        """Answer typed text (from the HUD) as if it had been spoken.

        Barges in on any in-flight turn, then runs the reply on its own thread
        so the caller (the UI) never blocks. Reuses the exact same brain +
        safety + speaker path as a voice turn.
        """
        text = (text or "").strip()
        if not text or self._state.shutting_down or self.brain is None:
            return
        with self._turn_lock:
            if not self._prepare_for_turn():
                log.info("Busy — dropping typed input %r", text[:40])
                return
            cancel = threading.Event()
            self._cancel_event = cancel

            def worker() -> None:
                try:
                    self._answer(text, cancel)
                except Exception:
                    log.exception("Typed turn failed")
                    self._speak_error()

            thread = threading.Thread(target=worker, name="ada-turn-text", daemon=True)
            self._turn_thread = thread
            thread.start()

    def _answer(
        self, text: str, cancel: threading.Event, timer: TurnTimer | None = None
    ) -> None:
        """Answer one piece of user text: local intents, else stream the reply.

        Shared by the voice path (`_run_turn`, which passes its timer with the
        listen/transcribe stages already marked) and typed input
        (`submit_text`, which starts timing here).
        """
        state = self._state
        if timer is None:
            timer = TurnTimer(publish=self._cfg.ui.show_timings)
        if self._console is not None:
            self._console.print_user(text)
        ui_events.publish("user_text", text)

        # Barge-in / shutdown may have landed between transcription and here;
        # bail before firing any local-intent side effects.
        if cancel.is_set():
            self._finish_turn()
            return
        if self._handle_local_intent(text):
            return

        assert self.brain is not None
        state.set_status(Status.THINKING)
        # Snapshot the Speaker generation so sentences streamed by this turn
        # can't leak into a later turn if we are barged in mid-reply.
        self._turn_gen = (
            self.speaker.current_generation() if self.speaker is not None else None
        )
        spoke = False

        def on_sentence(sentence: str) -> None:
            nonlocal spoke
            if cancel.is_set():
                return
            if not spoke:
                spoke = True
                timer.mark("think")
                state.set_status(Status.SPEAKING)
            ui_events.publish("assistant_delta", sentence)
            if self.speaker is not None:
                self.speaker.enqueue(sentence, self._turn_gen)

        full_text = self.brain.handle(text, on_sentence=on_sentence, cancel=cancel)
        if self.speaker is not None and not cancel.is_set():
            if not self.speaker.wait_until_done(timeout=_SPEAK_TIMEOUT_S):
                log.warning("Gave up waiting for playback after %.0fs", _SPEAK_TIMEOUT_S)
                self.speaker.stop()
        if spoke:
            timer.mark("speak")
        clean = full_text.strip()
        if self._console is not None and clean:
            self._console.print_assistant(clean)
        ui_events.publish("assistant_done", clean)
        if not cancel.is_set():
            timer.finish()
        self._finish_turn()

    def _speak_error(self) -> None:
        """Common failure path: flag ERROR, apologize out loud (best effort)."""
        state = self._state
        state.set_status(Status.ERROR)
        chime_error(self._cfg.ui.chimes)
        ui_events.publish("assistant_done", "")
        try:
            if self.speaker is not None and not state.shutting_down:
                self.speaker.enqueue("Sorry, something went wrong.")
                self.speaker.wait_until_done(timeout=_ERROR_SPEAK_TIMEOUT_S)
        except Exception:  # noqa: BLE001 - best effort only
            log.exception("Could not speak the error message")
        self._finish_turn()

    def _finish_turn(self) -> None:
        """Return the shared state to its resting status after a turn."""
        state = self._state
        if state.shutting_down:
            return
        muted = state.muted
        state.set_status(Status.MUTED if muted else Status.IDLE)
        # Back at rest with no recording active: flush any timer/reminder
        # announcements that arrived mid-turn (skip while muted).
        if not muted:
            self._flush_announcements()

    def _flush_announcements(self) -> None:
        """Speak any announcements buffered while a recording was active."""
        speaker = self.speaker
        if speaker is None:
            return
        with self._announce_lock:
            pending = self._pending_announcements
            self._pending_announcements = []
        for text in pending:
            speaker.enqueue(text)

    # -- local intents (no LLM round trip) -------------------------------------
    def _handle_local_intent(self, text: str) -> bool:
        """Handle cancel / mute / shutdown phrases locally. True if handled."""
        cfg = self._cfg
        assert self._confirmer is not None and self._audit is not None
        phrase = _normalize(text)

        if phrase in _CANCEL_PHRASES:
            log.info("Local intent: cancel")
            chime_done(cfg.ui.chimes)
            self._finish_turn()
            return True

        if phrase in _MUTE_PHRASES:
            log.info("Local intent: mute")
            self._audit.record("mute", muted=True, via="voice")
            self._say_and_wait("Muting. Unmute from the Ada window or menu bar.")
            self._state.set_muted(True)
            return True

        shutdown_phrases = set(_SHUTDOWN_PHRASES)
        name = cfg.assistant.name.strip().lower()
        if name:
            shutdown_phrases.add(f"goodbye {name}")
        if phrase in shutdown_phrases:
            log.info("Local intent: shutdown")
            approved = self._confirmer.confirm(
                title=f"Quit {cfg.assistant.name}?",
                message=f"You said “{text.strip()}”. Shut down the assistant?",
            )
            if not approved:
                log.info("Shutdown declined")
                self._finish_turn()
                return True
            self._say_and_wait("Goodbye.")
            self._state.request_shutdown()
            return True

        return False

    # -- helpers ----------------------------------------------------------------
    def _say_and_wait(self, text: str, timeout: float = 30.0) -> None:
        """Speak one phrase and block until playback ends (best effort)."""
        if self.speaker is None:
            return
        self._state.set_status(Status.SPEAKING)
        self.speaker.enqueue(text)
        self.speaker.wait_until_done(timeout=timeout)

    def _announce(self, text: str) -> None:
        """Timer/reminder announcer wired into the productivity tools.

        Safe to call at any time, including before build() has produced a
        Speaker (it just logs in that case). While a recording is active (or
        we are mid-turn), the announcement is buffered and flushed on return
        to IDLE so it never plays over — and gets recorded into — the
        utterance. The timer chime itself is played separately by
        productivity._fire_timer, so a buffered announcement is not silent.
        """
        speaker = self.speaker
        if speaker is None:
            log.info("Announcement (no speaker available): %s", text)
            return
        status = self._state.status
        if self._recording.is_set() or status in (
            Status.LISTENING,
            Status.THINKING,
        ):
            with self._announce_lock:
                self._pending_announcements.append(text)
            return
        speaker.enqueue(text)


# -- text REPL (`ada text`) --------------------------------------------------
def run_text_repl(cfg: Config, speak: bool = False) -> None:
    """Interactive typed conversation with the same brain and tools.

    No microphone, wake word, VAD, or STT is built — text mode works even if
    the audio dependencies are broken. Confirmation prompts appear in the
    terminal instead of native dialogs. 'exit' / 'quit' (or Ctrl+C / Ctrl+D)
    leaves the REPL.
    """
    from rich.console import Console

    state = AssistantState()
    assistant = Assistant(cfg, state, console=None)
    assistant.build(voice=False, tts=speak, confirmer=ConsoleConfirmer())
    state.set_status(Status.IDLE)
    assert assistant.brain is not None
    # Prime the local model (weights + prompt-prefix KV cache) while the user
    # types their first message, same as `ada run` does.
    threading.Thread(target=assistant.warmup, name="ada-warmup", daemon=True).start()

    console = Console(highlight=False)
    name = cfg.assistant.name
    console.print(
        f"[bold cyan]{name}[/] text mode — type a message; 'exit' to leave."
    )

    try:
        while True:
            try:
                line = input("you> ")
            except EOFError:
                break
            line = line.strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit"):
                break

            first = True

            def on_sentence(sentence: str) -> None:
                nonlocal first
                if first:
                    console.print(f"[bold green]{name.lower()}>[/] ", end="")
                    first = False
                console.print(sentence, end=" ", soft_wrap=True)
                if speak and assistant.speaker is not None:
                    assistant.speaker.enqueue(sentence)

            try:
                assistant.brain.handle(line, on_sentence=on_sentence, cancel=None)
            except Exception:
                if not first:
                    console.print()
                    first = True
                log.exception("Text turn failed")
                console.print("[red]Something went wrong — see the log for details.[/red]")
                continue
            if not first:
                console.print()  # terminate the streamed reply line
            if speak and assistant.speaker is not None:
                assistant.speaker.wait_until_done(timeout=_SPEAK_TIMEOUT_S)
    except KeyboardInterrupt:
        console.print()
    finally:
        assistant.shutdown()
        state.request_shutdown()
