# Ada — Project Handoff

> A single document to bring a fresh Claude chat (or a new developer) fully up to speed on this project: what it is, everything that's been done, the current state, and **why** each design decision was made. Read this top to bottom and you can pick up exactly where things left off.

**Last updated:** 2026-07-14
**Project root:** `/Users/anthony/Documents/AI assistant` *(note the space in the path — always quote it in shell commands)*
**Status in one line:** Fully built, reviewed, and tested (**292 tests passing**, live-verified against a running Ollama). The brain is **pluggable and local by default** (Ollama + `gemma4:12b`) — free, on-device. v0.2 shipped: a **liquid-glass HUD with rich graphical answer cards**, a **~5× latency cut** (simple turns ≈1.5 s warm; thinking-mode disabled + prompt-cache warmup + pipelined TTS), **38 tools** (music, calendar, browser tabs, screen context, confirmed screenshot/typing, clipboard, on-screen display), and **persistent long-term memory + usage-pattern learning**. Run: install Ollama, `ollama pull gemma4:12b`, then `ada run`.

---

## 0. TL;DR for the next session

- **What it is:** "Ada", a privacy-conscious, on-device-first voice assistant for macOS (Apple Silicon). Say something → it answers out loud, **shows the answer graphically on screen**, or does it (opens apps, controls music/calendar/browser/windows/volume, finds files, sets reminders/timers, searches the web, remembers things about you).
- **Language/stack:** Python 3.13, `src/` layout, package name `ada`, CLI command `ada`. Runs in a virtualenv at `.venv`.
- **UI:** a native on-screen **liquid-glass HUD** is the default surface (pywebview/WKWebView) — layered blur glass with specular edges and an aurora glow, an animated orb, a live transcript, **rich display cards** (tables, bar/line charts, lists, key-values, images, web results, files, now-playing, live countdown timers, calendar), tool-activity chips, and a per-turn "⚡ 1.8s" latency readout. Talk (tap the orb / ⌘⇧A), **type**, mute, or quit. Alternatives: `--menubar` (rumps icon) and `--no-gui` (terminal). Needs the `gui` extra; falls back gracefully if pywebview is missing.
- **Pipeline:** push-to-talk hotkey (⌘⇧A) → Silero VAD endpointing (600 ms end-silence) → **local Whisper STT** (`small.en`; Deepgram cloud optional) → **pluggable brain: local Ollama model (default `gemma4:12b`, thinking mode OFF) streaming tool loop** (**38 macOS tools**; cloud Claude optional via `llm.provider: anthropic`) → **natural local Kokoro neural voice, pipelined** (next sentence synthesizes while the current one plays; macOS `say` / ElevenLabs optional). **Fully local and free by default.** Warm simple turns ≈1.5 s; tool turns 3–8 s.
- **Learning:** long-term memory in `data/memory.jsonl` (plain, user-inspectable JSONL; git-ignored) — "remember that…" persists facts, top-matching facts are auto-injected into each request, `usage_patterns` mines the audit log for habits.
- **This is NOT a git repository.** There are no commits. Section 1 documents the work as chronological phases instead. If you want version control, `git init` — a `.gitignore` is already in place (it excludes `.env`, `logs/`, `data/`, `.venv/`, caches).
- **To run:** install **Ollama** (from ollama.com) and pull a model (`ollama pull gemma4:12b`), then `ada run`. No API key needed — the brain is local. *(The old $0-Anthropic-credit blocker no longer affects running; Anthropic is now just an optional cloud provider — see §5.)*
- **Verify current state on arrival:**
  ```bash
  cd "/Users/anthony/Documents/AI assistant"
  .venv/bin/python -m pytest -q      # expect: 292 passed
  .venv/bin/ada doctor               # expect: all green (checks the Ollama server + model when the brain is local)
  ```

---

## 1. Work log (what has been done, in order)

Not git commits — a chronological record of the phases. Each phase was substantial.

### Phase 1 — Scaffold & contracts
Read the original brief (`/Users/anthony/Downloads/AI_assistant_project_prompt.md`), probed the environment (macOS 26, Apple Silicon, Python 3.13), and hand-wrote the **stable contracts** that everything else builds against: `config.py` (+ `config.default.yaml`), `state.py` (thread-safe `AssistantState`/`Status`), `log.py`, `tools/registry.py` (the `@tool` decorator + `ToolSpec`), and the provider Protocols (`stt/base.py`, `tts/base.py`, `safety/base.py`). These were written by hand deliberately so the parallel build below had a fixed target to code against.

### Phase 2 — Parallel build (8 + 2 agents)
Installed all dependencies into `.venv`, pre-downloaded the models (openWakeWord phrases, Silero VAD, Whisper `small.en`), then ran a multi-agent workflow that implemented the eight subsystems in parallel (audio, stt, tts, brain, tools, safety, ui, docs), followed by an integration stage (main event loop + CLI, and the pytest suite). Result: 47 files, the app booting, 19 tools registering, first tests passing.

### Phase 3 — Adversarial review (37 agents)
Ran a 6-lens review workflow (concurrency, Anthropic-API usage, macOS integration, security, voice-pipeline logic, cross-module wiring). Each finding was then independently **verified by a separate agent that tried to refute it**. Outcome: **30 confirmed real bugs** (1 refuted) that the unit tests had not caught.

### Phase 4 — Fixes (4 parallel agents)
Fixed all 30 findings across four disjoint file groups (brain, tts, loop/state/audio, tools/security). Re-ran the full suite (green) plus a mocked end-to-end brain turn. See Section 6 for the notable bugs and why the code looks the way it does because of them.

### Phase 5 — Rename Jarvis → Ada
At the user's request, renamed the assistant. This was a full rename, not cosmetic: package directory `src/jarvis` → `src/ada`, CLI command `jarvis` → `ada`, logger namespace `jarvis.*` → `ada.*`, thread names, `pyproject.toml`, tests, docs, and the `JARVIS_CONFIG` env var → `ADA_CONFIG`. Two things were deliberately **protected** from the rename: the `hey_jarvis` openWakeWord model id (an external asset — kept as one wake-phrase option), and the "in the spirit of Jarvis" personality line (rephrased to "the manner of a calm, capable, lightly witty butler" so it wasn't a nonsensical self-reference). The default trigger was also switched to push-to-talk (see decision D3). Docs were rewritten to describe the hotkey-first reality accurately (no false "just say Hey Ada" claims). Full suite green under the new name.

### Phase 6 — API keys installed
User supplied an Anthropic key and a Deepgram key. Stored them in `.env` (perms `600`, git-ignored, never logged), briefly switched `stt.provider` to `deepgram`, and validated both: **Deepgram works**; the **Anthropic key authenticates but the account has $0 credit** (this was the blocker at the time — no longer relevant now that the brain and STT default to local; both keys remain as optional cloud fallbacks, see §5).

### Phase 7 — v0.2: speed, graphics, control, learning (5 parallel agents + hand-written contracts)
User feedback: "slow, can't do much, doesn't show much on screen; make it glassy-futuristic; give it app access; make it learn." Benchmarked first (M3 Pro, 18 GB): the #1 latency source was that **gemma4:12b is a thinking model** — ~100 silent reasoning tokens (~6–7 s at 16 tok/s) before any output; #2 was warmup neither priming the prompt-prefix KV cache (~8 s of tool-schema eval on the first turn) nor loading the model with matching `num_ctx` (forcing a reload). The orchestrator hand-wrote the cross-cutting contracts (display-card schema in `ui/events.py`, `ToolResult` in the registry, tool/timing events published from `brain/base.py`, `perf.TurnTimer`, config additions), then five agents built in parallel on disjoint files: (1) the liquid-glass HUD v2 with all display-card renderers; (2) brain latency — `think: false` with graceful fallback + cache-priming warmup + system-prompt update; (3) 15 new macOS tools + display cards retrofitted onto 6 existing tools; (4) the learning layer (`brain/knowledge.py` + memory tools); (5) pipelined TTS (synthesize N+1 while N plays) + early first-clause emission. Integration wired `_SUBMODULES`, knowledge injection, and TurnTimer into the turn loop. **292 tests passing**; live-verified against real Ollama (simple turn 1.5 s warm; `remember` → persisted facts → recalled next turn; `show_chart` produced a real chart card).

---

## 2. How to run, test, and verify

All commands assume you are in the project root (`cd "/Users/anthony/Documents/AI assistant"`). The `ada` command lives at `.venv/bin/ada`.

```bash
.venv/bin/ada doctor        # health check: deps, mic, models, keys (masked), say/osascript
.venv/bin/ada text          # typed REPL — talk to the brain without a mic (fastest test)
.venv/bin/ada text --speak  # same, but also speaks replies aloud
.venv/bin/ada run           # the real assistant: on-screen HUD window (default) + push-to-talk (tap the orb / ⌘⇧A, or type)
.venv/bin/ada run --menubar      # macOS menu-bar icon instead of the HUD
.venv/bin/ada run --no-gui       # terminal only (console status; --no-menubar is a hidden back-compat alias)
.venv/bin/ada say "hello"    # test the configured TTS voice
.venv/bin/ada listen         # record one utterance with VAD and print the transcript
.venv/bin/python -m pytest -q   # 292 tests, ~2s, fully offline (no mic/network/keys needed)
```

`--config PATH` works on every subcommand. Config is auto-discovered: explicit `--config` → `$ADA_CONFIG` → `./config.yaml` → `~/.config/ada/config.yaml`.

---

## 3. Architecture

### Component flow
```
mic (sounddevice callback thread)
  → frame queue (1280-sample int16 frames @ 16 kHz)
    → worker loop (Assistant.run): push-to-talk hotkey OR optional wake word
      → UtteranceRecorder (Silero VAD endpointing: pre-roll + speech + end-silence)
        → STT (faster-whisper local, default / Deepgram REST optional)
          → Brain (pluggable: local Ollama, default / cloud Anthropic — streaming tool loop)
            ├─ SentenceSplitter → Speaker queue → TTS (kokoro / say / ElevenLabs)   [speak as it streams]
            └─ tool_use → Policy (allow/deny/confirm) → Confirmer (native dialog) → tool handler → AuditLog
State (thread-safe status + mute + shutdown)  ←→  UI surface (HUD default / menu bar / console) + hotkey listener
UI event bus (user_text / assistant_delta / assistant_done / display / tool / timing)  →  UI surface [the HUD renders
    display payloads as glass cards, tool payloads as activity chips, timing payloads as the ⚡ latency readout]
Tools → ToolResult(speech, display card)  →  brain/base publishes "display"; the model itself can push cards via
    the show_on_screen / show_chart tools
KnowledgeStore (data/memory.jsonl)  →  top-K facts auto-injected into each user message (bracketed prefix,
    alongside the timestamp);  remember/recall/forget/usage_patterns tools share the same store instance
perf.TurnTimer  →  listen / transcribe / think / speak marks per turn  →  "timing" event + log
Typed input:  HUD text box → Assistant.submit_text() → shares _answer() (brain + tools + safety + speaker) with the voice path
```

### Threading model (important — several bugs lived here)
- **Mic callback thread** (sounddevice) → pushes frames into a `queue.Queue`.
- **Worker loop** (`Assistant.run`) → drains the queue, checks the push-to-talk activation Event / wake detector, and spawns turns. Runs on a daemon (worker) thread when the HUD or menu bar owns the main thread, or on the main thread with `--no-gui`.
- **Per-turn thread** (`_run_turn`, or `ada-turn-text` for typed input via `submit_text`) → records + transcribes + calls the brain + speaks, one per activation, so the worker loop keeps scanning for barge-in. Carries a `cancel` `threading.Event`.
- **Speaker worker thread** → serializes TTS playback of streamed sentences.
- **HUD window (pywebview)** → the default UI; `webview.start()` **blocks the process main thread** until the window closes, so the assistant loop runs on a worker thread. Status changes (`AssistantState` listeners) and transcript events (the UI event bus) are pushed into the WKWebView via `window.evaluate_js`; the front-end calls back through `window.pywebview.api` (`ready`/`talk`/`sendText`/`toggleMute`/`quit`). Same main-thread-owns-the-UI pattern the menu bar uses.
- **rumps menu bar** → alternative UI (`--menubar`); must run on the **process main thread** (macOS requirement); a `rumps.Timer` polls `AssistantState` and updates the icon.
- **Cancellation / barge-in** is coordinated entirely through `threading.Event`s and a Speaker "generation" counter (see decision D9).

### Project structure
```
pyproject.toml            # package metadata, extras: gui (HUD/pywebview) / menubar / hotkey / websearch / full
config.yaml               # ACTIVE user config (llm.provider = ollama, stt.provider = local). Overrides bundled defaults.
.env                      # OPTIONAL cloud keys (ANTHROPIC_API_KEY, DEEPGRAM_API_KEY). perms 600, git-ignored.
.gitignore                # excludes .env, logs/, .venv/, caches, egg-info
README.md                 # user-facing guide
docs/DESIGN.md            # full engineering design doc (tool comparisons, mermaid diagrams, roadmap, risks)
HANDOFF.md                # this file
src/ada/
  __init__.py
  cli.py                  # argparse CLI: setup/doctor/run/text/say/listen. Heavy imports are lazy.
  main.py                 # Assistant class + the voice event loop + run_text_repl()
  config.py               # dataclass config + layered loader (defaults → yaml → .env secrets)
  config.default.yaml     # bundled defaults (copied by `ada setup`)
  state.py                # AssistantState (thread-safe status/mute/shutdown), Status enum, icons
  perf.py                 # TurnTimer — per-turn stage waterfall, publishes "timing" events
  log.py                  # rotating file log + console handler
  audio/                  # lazy __getattr__ package (so text mode never imports sounddevice)
    capture.py            # MicStream (sounddevice InputStream → queue)
    wakeword.py           # WakeWordDetector (openWakeWord, optional)
    vad.py                # UtteranceRecorder (Silero VAD endpointing)
    chime.py              # afplay system sounds (non-blocking)
  stt/
    base.py               # STTProvider protocol, STTResult
    local_whisper.py      # LocalWhisperSTT (faster-whisper, on-device)
    deepgram.py           # DeepgramSTT (REST via httpx, no SDK)
    __init__.py           # create_stt() with graceful fallback to local
  tts/
    base.py               # TTSProvider protocol (speak / stop / arm)
    say.py                # SayTTS (macOS /usr/bin/say)
    kokoro.py             # KokoroTTS (local neural voice, onnxruntime; ~340MB model in ~/.cache/ada/kokoro)
    elevenlabs.py         # ElevenLabsTTS (streaming PCM via httpx, no SDK)
    speaker.py            # Speaker: sentence-queue playback, generation-scoped barge-in
    __init__.py           # create_tts() with graceful fallback to say
  brain/
    prompts.py            # build_system_prompt() — STATIC per process (cache-friendly); screen/memory guidance
    memory.py             # ConversationMemory (bounded, time-expiring, tool-pair-safe trim)
    knowledge.py          # KnowledgeStore — persistent long-term facts (JSONL, keyword recall, atomic writes)
    sentences.py          # SentenceSplitter (streaming → speakable chunks; early first-clause emission)
    base.py               # BaseBrain: safe tool execution (policy → confirm → timeout → audit), ToolResult unwrap,
                          #   "tool"/"display" event publishing, optional .knowledge store
    ollama_client.py      # OllamaBrain (DEFAULT): /api/chat streaming + tools, think:false (+fallback),
                          #   cache-priming warmup, knowledge injection into the user-message prefix
    client.py             # AnthropicBrain: cloud Claude Messages API streaming tool loop (opt-in)
    __init__.py           # create_brain(cfg, ...) factory — selects the backend from llm.provider
  tools/                  # 38 tools across 14 modules
    registry.py           # @tool decorator, ToolSpec, ToolError, ToolResult(speech, display), anthropic_tools()
    __init__.py           # load_all(cfg) + get_config(); imports all tool modules (_SUBMODULES)
    applescript.py        # run_applescript() + quoting/date helpers (permission-error mapping)
    apps.py               # open_app / quit_app / open_url / open_path
    files.py              # search_files (mdfind) → files card
    system.py             # set_volume / mute_audio / lock_screen / system_sleep / battery / open_settings
    windows.py            # list_windows / window_action (System Events snapping)
    productivity.py       # create_reminder / create_note / set_timer (live timer card) / cancel_timers
    web.py                # search_web (ddgs) / fetch_url (SSRF-guarded) → web_results/markdown cards
    display.py            # show_on_screen (markdown card) / show_chart (bar|line) — the model's own screen access
    music.py              # music_control / now_playing (Spotify if running, else Apple Music)
    calendar_mac.py       # calendar_events (whose-clause bounded, cap 15) / create_event (confirm)
    browser.py            # browser_tabs / current_page / switch_tab / close_tab (confirm) — Safari/Chrome
    context.py            # front_app_info / capture_screen (confirm; screencapture→sips→data-URI card)
    input_text.py         # type_text (confirm; System Events keystroke, exact text in the dialog)
    clipboard.py          # read_clipboard / set_clipboard (pbpaste/pbcopy)
    memory_tools.py       # remember / forget (confirm) / recall / usage_patterns (+ get_store() shared instance)
  safety/
    base.py               # Decision, PermissionPolicy, Confirmer protocols
    permissions.py        # Policy: allowlists, risk tiers, fail-closed
    confirm.py            # DialogConfirmer (native osascript dialog) / ConsoleConfirmer
    audit.py              # AuditLog (JSONL, never stores audio or secrets)
  ui/
    events.py             # UI event bus: thread-safe pub/sub — transcript events + display/tool/timing;
                          #   its docstring is the CANONICAL display-card schema (11 card kinds)
    console.py            # ConsoleUI (rich status lines, banner)
    menubar.py            # run_menubar (rumps, main thread) — alternative surface
    hotkey.py             # HotkeyListener (pynput global hotkey)
    hud/                  # native HUD (DEFAULT surface)
      app.py              # run_hud (pywebview window) + _AdaApi (window.pywebview.api JS bridge); subscribes to state + event bus
      index.html          # self-contained front-end (inline CSS/Canvas/JS, no external assets; packaged via package-data)
      __init__.py         # hud_available() / run_hud re-exports
    __init__.py
tests/                    # 292 tests, offline, no hardware (per-module tool tests, knowledge store,
                          #   HUD event routing, Ollama think/warmup, speaker pipelining, early-chunk splitter)
```
Plus `data/memory.jsonl` (created at runtime): the long-term memory — plain JSONL, git-ignored, user-editable.

---

## 4. Design decisions and the reasoning behind them

This is the part that saves a new session the most time. Each decision includes the *why* so you don't re-derive or accidentally undo it.

**D1 — Python + `src/` layout, single-developer prototype.**
The brief asked for something a single developer can build and test; Python has the richest ecosystem for every piece of this pipeline (audio, wake word, VAD, Whisper, the Anthropic SDK). `src/` layout keeps imports honest (you test the installed package, not the working dir). A native/Swift app is noted as future work (v1.0) in the roadmap, not the MVP.

**D2 — Local-first pipeline for privacy.**
Wake-word detection (openWakeWord) and voice-activity detection (Silero) run entirely on-device; nothing leaves the machine while idle. Speech-to-text defaults to local Whisper, and the **LLM brain now defaults to a local model via Ollama** (see D11) — so the whole pipeline is on-device and free out of the box. Cloud services (Anthropic, Deepgram, ElevenLabs) are strictly opt-in and every provider fails *gracefully back to local* if its key/server is missing. This directly implements the brief's "avoid continuously sending room audio to the cloud."

**D3 — Push-to-talk (⌘⇧A) is the DEFAULT trigger; hands-free wake is optional.**
Critical and non-obvious: **openWakeWord ships no "hey ada" model.** Its only pretrained phrases are `hey_jarvis`, `hey_mycroft`, `alexa`, `hey_rhasspy`. So a name-matching hands-free wake word would require training a custom model (a real multi-hour offline job: synthetic-speech data generation + training). Rather than ship a mismatched trigger, the default is a push-to-talk hotkey (`hotkey.enabled: true`, combo `<cmd>+<shift>+a`): tap it, then speak; Silero VAD ends the utterance on silence. Hands-free is available two ways, both documented: (a) `wake.enabled: true` with a prebuilt phrase (default `hey_mycroft_v0.1` — note the spoken phrase then won't match "Ada"), or (b) train a custom "hey ada" model (roadmap). **Do not "fix" this by claiming a `hey_ada` model exists.** The hotkey needs macOS **Input Monitoring** permission (on by default now).

**D4 — Cloud-brain (Anthropic) model params: `claude-opus-4-8`, `effort: low`, streaming, no thinking/sampling params.**
*(Applies only to the optional cloud path, `llm.provider: anthropic`; these keys now live under `llm.anthropic.*` — the old flat `llm.model`/`llm.effort`/`llm.max_tokens` are gone. See D11.)* `effort: low` keeps voice replies snappy (short conversational answers). The request deliberately sends **no** `temperature`/`top_p`/`top_k` and **no** `thinking` parameter — these are removed on this model and would 400. Responses **stream** and are split into sentences so TTS starts before the full reply is generated (the single biggest perceived-latency win). See `brain/client.py`.

**D5 — Prompt caching via a static system prompt + time in the user message.**
The system prompt is built once per process and never changes (no timestamps), so the Anthropic prompt cache stays warm. The current time reaches the model as a bracketed prefix on each *user* message (`memory.py`), not in the system prompt. There's also a cache breakpoint on the last message of the history (system prompt + tools alone are below the 4096-token minimum cacheable prefix for this model, so the conversation-level breakpoint is what actually caches once history grows). See decision context in Section 6, bug #23.

**D6 — Tools are a registry of typed, risk-tiered functions.**
Each tool = an Anthropic JSON schema + a Python handler + a risk level (`safe` / `confirm`) + an optional human-readable call summary for confirmation dialogs. 19 tools across apps/files/system/windows/productivity/web. Handlers return short **speakable** strings and raise `ToolError` for expected failures. macOS control prefers native APIs (AppleScript / System Events / `open` / `mdfind` / `pmset`) over fragile screen-coordinate automation, per the brief.

**D7 — Safety model: allowlists + risk tiers + confirmation + audit + kill switch.**
- **Allowlists** in config: which folders (`file_roots`), which app dirs, which URL schemes, which apps are blocked.
- **Risk tiers:** `confirm`-risk tools (quit app, sleep) and anything in `always_confirm` trigger a **native macOS confirmation dialog** before running.
- **Policy fails closed:** any internal error in evaluation denies the action.
- **Audit log** (JSONL): every tool call (allowed/denied/confirmed/ok/error/duration) — but **never raw audio and never secrets**. Transcripts log only the recognized command text.
- **Kill switch / mute:** menu-bar mute, "stop listening" voice intent, and Ctrl+C / menu Quit.
- **Deliberately NOT implemented** (and why): arbitrary shell execution, file deletion, and raw mouse control — high-blast-radius with weak allowlisting. This is a conscious scope boundary, not an omission. Don't add them without a matching safety design.
- **Added in v0.2 WITH their matching safety design:** `type_text` (keyboard input) and `capture_screen` (screen capture) are both risk `confirm` — a native dialog shows the *exact* text to be typed / announces the screenshot before anything happens, every call is audited, and the screenshot is a temp file deleted immediately after being encoded for the HUD (never persisted). This is the pattern for any future high-blast-radius tool: confirm-tier + full disclosure in the dialog + audit.

**D8 — Secrets only in `.env`, never in YAML or source.**
`config.py` reads `ANTHROPIC_API_KEY` / `DEEPGRAM_API_KEY` / `ELEVENLABS_API_KEY` from the environment / a `.env` next to the config. YAML never contains secrets; the loader never writes secrets to disk. `.env` is `chmod 600` and git-ignored.

**D9 — Barge-in correctness via a Speaker "generation" counter.**
When the user interrupts (barge-in), `Speaker.stop()` bumps a generation number and drains the queue. Each turn captures the generation at its start and tags every enqueued sentence with it; a sentence from a cancelled turn carries a stale generation and is dropped rather than played during the next turn. TTS providers expose `arm()` (clear interrupt) which the Speaker calls *under its lock* atomically with the liveness check, closing a TOCTOU race where a `stop()` could be lost. This is why the TTS code has an `arm()` method and a generation parameter that look unusual — they exist to make barge-in race-free (Section 6, bugs #2/#3/#17/#18/#20/#26).

**D10 — Local intents handled before the LLM.**
"stop"/"cancel", "mute"/"go to sleep", "shut down"/"goodbye ada" are matched locally (exact normalized phrase, not substring — so "stop the music" is NOT swallowed as a cancel) and handled without a model round trip.

**D11 — Pluggable LLM brain via `llm.provider`; local (Ollama) is the DEFAULT.**
The brain is now provider-swappable, mirroring STT/TTS. Two backends share one safe-execution path (`brain/base.py` `BaseBrain`: permission policy → confirm → timeout → audit, plus the bounded tool loop) and are chosen by a `create_brain(cfg, …)` factory in `brain/__init__.py`:
- **`ollama` (DEFAULT)** — `brain/ollama_client.py` `OllamaBrain`, a local model served by Ollama at `http://localhost:11434` via the native `/api/chat` (streaming + tool calling). Free, no key, fully on-device. Config under `llm.ollama` (`model: gemma4:12b`, `host`, `temperature`, `num_ctx`, `keep_alive`).
- **`anthropic`** — `brain/client.py` `AnthropicBrain`, the original cloud Claude path, now opt-in (`llm.provider: anthropic`, needs `ANTHROPIC_API_KEY`). Config under `llm.anthropic` (`model: claude-opus-4-8`, `effort`, `max_tokens`).

Shared knobs (`max_tool_iterations`, `history_max_messages`, `reset_history_after_min`) stay at the `llm` top level. The old flat `llm.model`/`llm.effort`/`llm.max_tokens` keys are **gone** (moved under `llm.anthropic`).

**Honest tradeoff:** local is **free, private, weaker/inconsistent at tool calling**; cloud Claude is **strongest at tools**. Gemma is great at conversation, but some Gemma builds don't expose Ollama's `tools` API at all — when a model can't do tools, Ada detects it, says so once ("this local model can chat but can't run Mac actions"), and continues chat-only. The most reliable local tool callers are **`qwen2.5`** and **`llama3.1`** (also free/on-device); users switch with one line, `llm.ollama.model`. To make the whole pipeline on-device and free, **STT was also moved back to local** (`stt.provider: local`, Whisper `small.en`); Deepgram stays as an optional cloud fallback.

**D12 — Native on-screen HUD via pywebview; a UI event bus decouples it from the pipeline.**
`ada run` now defaults to a floating, frameless, translucent, always-on-top Jarvis-cyan glass panel (`ui/hud/app.py`, front-end `ui/hud/index.html`) rendered by **pywebview** (WKWebView + pyobjc) — a real macOS window, not a browser tab. It shows a canvas-drawn animated orb that reacts to state (idle / listening / thinking / speaking / muted / error), a status pill, a live transcript, a text box, and tap-to-talk / mute / quit controls; it's draggable (bottom-right by default). Verified: the JS↔Python bridge handshake and DOM updates work end-to-end (no window is launched in tests).
- **Why pywebview over Qt/Swift:** a native window with web-quality visuals (the glow-heavy orb + glass panel) for far less code than hand-painting a Qt/Tkinter scene; the front-end is one offline `index.html` (inline CSS/Canvas/JS, no external assets). A fully native SwiftUI app stays a v1.0 roadmap item.
- **Decoupling via a UI event bus** (`ui/events.py`): the pipeline *publishes* transcript events (`user_text` / `assistant_delta` / `assistant_done`) and any UI *subscribes*; status still flows through the existing `AssistantState` listeners. The loop never knows which surface (HUD / menu bar / console) is attached, and a broken UI listener is logged, never fatal.
- **Main thread:** `webview.start()` blocks the process main thread, so — exactly like the menu bar — the assistant voice loop runs on a worker thread while pywebview owns the main thread.
- **Typed input path:** the HUD box calls `Assistant.submit_text()`, which barges in on any in-flight turn and reuses the shared `_answer()` logic — same brain, tools, safety, and speaker as speech. Three input routes now: speak (hotkey/wake), type, or click the orb.
- **Modes & fallback:** default = HUD (`ui.hud: true`, `ui.theme: jarvis-cyan`); `--menubar` = rumps icon; `--no-gui` = terminal (`--no-menubar` is a hidden back-compat alias). Needs the `gui` extra (`pip install -e ".[gui]"`, included in `full`); if pywebview isn't installed, `ada run` falls back to the menu bar (if available), else the terminal, and prints how to enable the HUD.
- **v0.2 redesign ("liquid glass"):** layered 34px-blur glass with a specular gradient border, a slow three-blob aurora (cyan/violet/ice) behind the content, a refined orb (fewer, finer particles + specular arc), 430×680 resizable window, `vibrancy=True` passed only when the installed pywebview supports it. `prefers-reduced-motion` disables the heavy animation. The front-end markdown renderer is DOM-built (no innerHTML of payload text) and `openUrl` (the JS bridge for card links) refuses non-http(s) schemes.

**D13 — Rich answers via display cards on the event bus; `ToolResult` carries them.**
"Show, don't read": every answer with structure should land on screen as a graphical card while the voice speaks a one-line summary. Mechanism: tool handlers may return `ToolResult(speech, display)` — `speech` goes to the model (and TTS), `display` is a JSON card published as a `"display"` event by `brain/base.py` (so it works identically for both brain providers and never enters model context). The card schema (11 kinds: markdown, table, list, keyvalue, chart, image, web_results, files, now_playing, timer, calendar) is documented ONCE, canonically, in the `ui/events.py` docstring — keep Python producers and the HUD renderer in sync with it. Cards with the same `id` replace in place (that's how the timer card ticks live). The model has its own screen access via the `show_on_screen`/`show_chart` tools, and the system prompt pushes it to use them for anything structured. Tool activity is streamed as `"tool"` events (running → ok/error) for the HUD's chips.

**D14 — The latency program (why "fast" is a design property, not a knob).**
Measured on this M3 Pro/18 GB before the work: simple turns 8–13 s. Four fixes, biggest first: (1) **thinking off** — gemma4:12b silently generates ~100 reasoning tokens (~6–7 s) per reply; every Ollama request now sends `"think": false`, with a `_ThinkUnsupported` fallback that retries once without the param and remembers (mirrors the tools fallback; matches only real Ollama error shapes). Stream-side, `message.thinking` deltas are never spoken or stored. (2) **Warmup that actually warms** — the old warmup posted empty messages with no options, so the first turn paid ~8 s of tool-schema prompt eval AND a model reload (mismatched `num_ctx`). Warmup now sends the byte-identical request shape (system + tools + same options + think) with `num_predict: 1`, priming the KV prefix cache; `ada text` warms up too (background thread). (3) **Pipelined TTS** — Kokoro exposes `synthesize()`/`play_prepared()`; a single prefetch thread synthesizes sentence N+1 while N plays, exactly one chunk ahead; liveness is still decided at PLAY time under the Speaker lock, so the D9 barge-in contract is intact (a stale prefetched synthesis is discarded, never played). Providers without the capability keep the old serial path. (4) **Tighter endpointing + earlier first words** — VAD end-silence 900→600 ms; the SentenceSplitter emits the FIRST chunk at a clause boundary (≥6 words) or after ≥12 buffered words, so speech starts before the first full sentence lands. Every turn is instrumented by `perf.TurnTimer` (listen/transcribe/think/speak) → log + `"timing"` event → the HUD's ⚡ readout (`ui.show_timings`). Result, live-verified: warmup ~35 s once at launch, then simple turns ≈1.5 s, tool turns 3–8 s (two model rounds at ~16 tok/s is the remaining floor).

**D15 — Long-term memory: a plain local file, injected via the user message, shared with the tools.**
`brain/knowledge.py` `KnowledgeStore`: one JSONL file (`data/memory.jsonl`, `memory:` config section), one fact per line, keyword-overlap recall with recency/uses tiebreaks, atomic writes, max_facts eviction — deliberately NO embeddings/cloud/DB so the user can open, edit, and delete their own memory file. Injection preserves D5's static system prompt: `context_for()` produces one dense line that rides in the SAME bracketed prefix as the timestamp on the user message (`[Mon 2026-07-14 09:30 | Remembered about the user: …] text`). `Assistant.build()` wires the exact same store instance into `brain.knowledge` that the `remember`/`recall`/`forget`/`usage_patterns` tools use (via `memory_tools.get_store()`), so a fact saved mid-conversation is recallable on the next turn. `usage_patterns` mines the existing audit log (tool counts, opened apps, busiest hours) rather than keeping a second tracking system.

---

## 5. Current state / what's needed to run

The brain is **local now**, so running no longer depends on any cloud account.

- **To run: install Ollama + pull a model.** Install Ollama from **ollama.com**, then `ollama pull gemma4:12b` (or another tag). Start Ollama (the app, or `ollama serve`) and it listens at `http://localhost:11434`. `ada doctor` checks the server is up and the model is pulled (❌ with an `ollama pull <model>` hint if not). The default `llm.provider: ollama` needs no API key.
- **STT is back to local.** `stt.provider: local` (Whisper `small.en`, already downloaded), so the whole pipeline is on-device and free. Deepgram remains a one-line opt-in (`stt.provider: deepgram`) that sends audio to the cloud.
- **Optional cloud keys still live in `.env`** (values not reproduced here for safety; perms 600, git-ignored; `ada doctor` shows them masked): `ANTHROPIC_API_KEY`, `DEEPGRAM_API_KEY`. They are **fallbacks only** now.
  - **Anthropic** matters only if you switch the brain back to cloud with `llm.provider: anthropic`. That account still has **$0 credit**, so the cloud brain would 400 with *"Your credit balance is too low"* until credit is added at **console.anthropic.com → Plans & Billing** — but this **no longer blocks running**, because the default brain is local.
  - **Deepgram** is valid; only used if you set `stt.provider: deepgram`.
- **Security note:** the keys were pasted into a chat. Consider rotating them (regenerate in the Anthropic/Deepgram consoles, replace in `.env`) if that conversation isn't private.

---

## 6. Notable bugs found & fixed (so you understand the "why" of odd-looking code)

The adversarial review found 30 real bugs; all were fixed. The ones most likely to confuse a future reader:

- **#1 History poisoning (major).** If a streamed response ends for a non-`tool_use` reason (e.g. `max_tokens` truncating a tool call) but still contains a `tool_use` block, the assistant turn would be stored with no matching `tool_result`, and *every* later request would 400. Fix: `brain/client.py` appends synthetic error `tool_result`s for any orphaned `tool_use` before returning. This is why that branch exists. Also raised default `max_tokens` 1024 → 2048 as defense-in-depth.
- **#22 Memory trim (major).** `ConversationMemory.trim()` protects the trailing in-flight `tool_use`/`tool_result` pair so it's never severed and history is never emptied mid-loop.
- **#2/#3/#17/#18/#20/#26 TTS barge-in races (major).** See decision D9 — the `arm()` method, the generation counter, and the sentinel-safe queue drain all exist to make interruption race-free.
- **#14/#4 Mic ownership (major).** Recording ownership is an explicit `threading.Event` (`_recording`), not inferred from status; `set_muted()` no longer stomps an active turn's status. Muting mid-turn now cancels the turn.
- **#11 SSRF (security).** `fetch_url` resolves the host and refuses loopback/private/link-local/reserved addresses, and re-checks on every redirect hop (redirects followed manually). Prevents prompt-injected web content from making Ada hit internal services.
- **#9/#10 App/path bypass (security).** `blocked_apps` is matched after canonicalizing `.app`/path forms; `open_path` requires confirmation for launchable/executable suffixes and cross-checks `.app` names against the block list.
- **#5/#6/#24 macOS automation-consent errors.** `-1743` ("Not authorized to send Apple events") is mapped to a friendly "grant Automation permission" message, and window/focus tools no longer mask it as "app isn't running".
- **#7 lock_screen.** Uses the private `login.framework SACLockScreenImmediate` (falls back to `pmset displaysleepnow`) so it actually locks regardless of the "require password" setting.
- **#13 Wake self-trigger (no-AEC mitigation).** Because there's no acoustic echo cancellation, while Ada is speaking the wake detector requires a higher confidence (`>= max(threshold, 0.7)`) so its own TTS is far less likely to self-trigger barge-in — while a clearly-spoken wake still crosses the bar. Documented as a known limitation.

---

## 7. Known limitations & honest expectations

- **No acoustic echo cancellation** → the wake self-trigger mitigation above is a partial fix, not perfect.
- **AppleScript / System Events brittleness** across macOS versions; window snapping is best-effort and multi-monitor uses a single-screen approximation unless `pyobjc` is present.
- **Whisper first-load latency** (model load on first `ada run`/`setup`); transcription itself is ~1s for short commands with `small.en`. Deepgram (optional cloud STT) is faster if you opt in.
- **LLM misinterpretation** of transcribed speech is mitigated by confirmations + allowlists, not eliminated.
- **macOS permissions** must be granted the first time: Microphone (recording), Input Monitoring (the hotkey), Accessibility (window management), Automation (Reminders/Notes/System Events).

---

## 8. Roadmap (from docs/DESIGN.md)

- **v0.1:** the original MVP (voice loop, 19 tools, HUD v1).
- **v0.2 (current):** ✅ shipped — liquid-glass HUD with display cards, the latency program (think:false, cache-priming warmup, pipelined TTS, tuned VAD), 38 tools (music/calendar/browser/context/screenshot/typing/clipboard/display), long-term memory + usage patterns.
- **v0.3:** vision — feed `capture_screen` output to a multimodal local model ("what am I looking at?" beyond metadata); streaming STT (Deepgram websockets); a follow-up window after a reply (a few seconds of trigger-free listening for "and also…"); **a custom-trained "hey ada" wake word**; proactive habit suggestions from usage patterns.
- **v1.0:** native Swift menu-bar app or a packaged `.app` (briefcase/py2app), auto-start login item.

---

## 9. Gotchas for the next Claude session

1. **The project path contains a space** — always quote it in shell commands.
2. **Not a git repo** — there is no history to inspect; this file is the history. Offer `git init` if the user wants it (`.gitignore` is ready).
3. **Run Python/tests via `.venv/bin/python` / `.venv/bin/ada`**, not the system Python (system Python has none of the deps).
4. **STT is local by default now** (`stt.provider: local`, Whisper `small.en`). Deepgram is a one-line opt-in (`stt.provider: deepgram`) that sends audio to the cloud.
5. **`llm.provider` defaults to `ollama`** — the brain is local (Ollama, default `gemma4:12b`). Tool calling varies by local model: **Gemma chats well but its Ollama tool support is inconsistent** — some builds don't support the `tools` API, so Ada detects it, says so once, and continues chat-only. For reliable Mac actions switch `llm.ollama.model` to **`qwen2.5`** or **`llama3.1`** (both local/free). The old flat `llm.model`/`llm.effort`/`llm.max_tokens` keys are gone — cloud settings now live under `llm.anthropic`.
6. **Anthropic is now OPTIONAL, not a blocker.** It's used only when `llm.provider: anthropic`. That account still has $0 credit, but that only matters if you switch back to the cloud brain — it does **not** block running the (default, local) pipeline.
7. **`hey_jarvis` is intentional** — it's one of openWakeWord's prebuilt phrase options, not a leftover of the old name. Don't rename it to `hey_ada` (that model doesn't exist).
8. **Anthropic API constraints (cloud path only, `llm.provider: anthropic`):** no `temperature`/`top_p`/`top_k`, no `thinking` param, use `output_config={"effort": ...}`, stream for large `max_tokens`. Sending the forbidden params 400s.
9. **A project memory already exists** at `~/.claude/projects/-Users-anthony-Documents-AI-assistant/memory/` (auto-loaded in this project's sessions) with a condensed version of this context.
10. **Tests are the safety net** — 292 of them, fully offline (mocked subprocess/network/audio). Run them after any change: `.venv/bin/python -m pytest -q`.
11. **The HUD needs the `gui` extra (pywebview).** `pip install -e ".[gui]"` (or `".[full]"`) installs it; without it `ada run` falls back to the menu bar or terminal and prints how to enable the HUD. `webview.start()` **blocks the process main thread**, so the assistant loop runs on a worker thread (same pattern the menu bar uses). The front-end is `ui/hud/index.html` — a self-contained page (inline CSS/Canvas/JS, no external assets) shipped as **package-data**, so don't move/rename it without updating packaging. Typed HUD input goes through `Assistant.submit_text()` → the shared `_answer()` path; the new tests live in `tests/test_hud.py` (event bus + `submit_text`/`_answer` + the JS-bridge object; no window is launched).
12. **Do NOT re-enable thinking** (`llm.ollama.think: true`) casually — on gemma4:12b it adds ~6–7 s of silent reasoning before every reply. It exists as a config escape hatch, not a quality knob for voice.
13. **The first launch after boot is slow on purpose elsewhere:** warmup takes ~35 s (model load + priming the 38-tool prompt prefix into Ollama's KV cache) so every later turn is fast. It runs on a background thread in both `ada run` and `ada text`. Don't "optimize" warmup back to an empty request — and keep its request body byte-identical in shape to real turns (same options/think/tools), or the cache prime and even the model load are wasted (mismatched `num_ctx` forces a reload).
14. **The display-card schema lives in ONE place** — the `ui/events.py` docstring. If you add a card kind or field, update that docstring, the producers (tools), and the HUD renderer (`ui/hud/index.html`) together. Cards never enter model context; anything the model needs must be in the tool's `speech`.
15. **Long-term memory is a plain file** — `data/memory.jsonl` (git-ignored). Deleting it wipes Ada's memory; it's user-editable by design. The tools and the brain share ONE store instance via `memory_tools.get_store()` — don't construct a second `KnowledgeStore` for the same path.
16. **38 tools is near the practical ceiling for a 12B local model** — descriptions are deliberately crisp and disjoint. If tool selection degrades after adding more, prune or merge before blaming the model; `qwen2.5`/`llama3.1` remain the most reliable local tool-callers.
