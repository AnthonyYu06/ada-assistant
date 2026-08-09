# Ada

A privacy-conscious voice assistant for your Mac with a futuristic on-screen HUD — Ada is a floating, glowing orb you can talk to, type to, and click. Tap the push-to-talk hotkey (**⌘⇧A**) or the orb, ask for something in plain English, and it answers out loud — or actually does it: opens apps, finds files, sets reminders and timers, manages windows, adjusts volume, and looks things up on the web. Hands-free wake-word detection is available as an option; voice-activity detection, speech recognition, and the LLM brain all run locally by default (no cloud, no API bill to run), and every action the assistant takes goes through a permission policy, an on-screen confirmation for anything risky, and an audit log. The HUD floats above your other windows and shows — in its live orb and transcript — exactly what Ada is doing; a menu-bar mode and a plain terminal mode are there when you want them.

## Features

- **A futuristic liquid-glass HUD**: `ada run` opens a floating, frameless glass panel — deep-blur layered glass with specular edges and a slow aurora glow — around a canvas-drawn animated **orb** that reacts to Ada's state (idle breathing, listening pulse, thinking shimmer, a speaking waveform, muted/error tints). It carries a status pill, a live chat transcript, a text box, tap-to-talk, and mute/quit controls. Always-on-top, draggable, resizable.
- **Answers you can see**: Ada doesn't just talk — it puts answers **on screen as glass cards**: tables, bar/line charts, lists, key-value panels, web results, found files, calendar events, a live now-playing tile, ticking timer cards with progress rings, screenshots, and formatted text. Tool activity shows as live chips ("⚙ Opening Safari… ✓"), and each reply gets a subtle "⚡ 1.8s" latency readout you can expand into a stage-by-stage waterfall.
- **Three ways in**: **speak** (hotkey or optional wake word), **type** in the HUD's box, or **click the orb** to talk. Typed input runs through the exact same brain, tools, and safety checks as speech.
- **Push-to-talk by default**: tap **⌘⇧A**, then just speak — Ada records until you stop talking (local Silero VAD endpointing) and replies. No wake word required.
- **Optional hands-free**: turn on a local wake word (openWakeWord) if you'd rather not touch the keyboard — no audio leaves your Mac while it waits. It ships with prebuilt phrases (e.g. "hey mycroft"); a name-matching "hey ada" phrase needs a custom-trained model.
- **Fast turns**: simple questions answer in ~1.5 s once warm. Thinking-mode is disabled on reasoning models (the silent multi-second "reasoning" phase is the #1 local-LLM latency killer), the full prompt prefix is pre-warmed into Ollama's cache at startup, VAD endpointing is tuned tight, the first clause of a reply is spoken while the rest is still generating, and the next sentence is synthesized while the current one plays.
- **A natural voice, free and local**: Ada speaks with **Kokoro**, a neural text-to-speech model that runs on your Mac via onnxruntime — no cloud, no API key, no cost. It sounds far more lifelike than the built-in macOS voice (the free, local answer to something like ElevenLabs). Pick from several voices (US/British, male/female) with one config line, or fall back to the macOS `say` voice or cloud ElevenLabs.
- **A real brain, local by default**: a pluggable LLM. Out of the box it runs a local model via Ollama (e.g. `gemma4:12b`) — free, on-device, no API key. Switch to cloud Claude (Anthropic API) with one config line if you prefer. Either way it uses tool calling to chain a few tool calls within a bounded loop. *(Local models vary at running Mac actions — see the tool-calling note below.)*
- **Does real things — 38 tools**: open apps/files/URLs, search your files, move and resize windows, set volume, create Reminders and Notes, set timers, answer questions from the web — plus control **music** (Spotify/Apple Music), read and create **calendar** events, list/switch/close **browser tabs**, tell you **what's on screen** (front app + current page), take a confirmed **screenshot** onto the HUD, **type text** for you (confirmed), and read/set the **clipboard**.
- **Remembers you, learns your habits**: say "remember that…" and the fact persists across sessions in a plain local file (no cloud, no embeddings — you can open and edit it). Relevant memories are recalled automatically on every request, and `usage_patterns` can chart when and how you actually use Ada from its own audit log.
- **Safe by design**: allowlisted app folders and file roots, native macOS confirmation dialogs for risky actions, a JSONL audit log of every tool call, and a mute/kill switch that is always one click away.
- **Interruptible**: tap **⌘⇧A** again while it is talking to cut it off and give a new command (barge-in). If you've enabled hands-free wake, saying the wake phrase during playback also interrupts.
- **Your choice of surface**: the HUD window by default, a lightweight macOS menu-bar icon (`--menubar`), or plain terminal (`--no-gui`) — each shows idle / listening / thinking / speaking / muted at a glance. Voice works in every mode.
- **Works without a mic too**: a typed REPL (`ada text`) for testing and quiet environments — or just type in the HUD box.

## Requirements

- macOS on Apple Silicon (developed on macOS 26; recent versions should work).
- Python 3.11 or newer (3.13 recommended).
- About **2 GB of free disk** for the speech models (whisper; plus a small wake-word model only if you enable hands-free).
- **Ollama** (from [ollama.com](https://ollama.com)) and a pulled local model for the brain (default `gemma4:12b`, ~8 GB) — budget **~5–20 GB of disk** depending on the model size for the model. No API key needed to run.
- For the on-screen HUD: **pywebview** (installed by the `gui` extra, and included in `full`). Without it, `ada run` falls back to the menu bar or terminal.
- For the natural Kokoro voice: the **`voice`** extra (`kokoro-onnx`, in `full`) plus a one-time **~340 MB** model download that `ada setup` handles. Without it, Ada uses the built-in macOS voice.
- *(Optional)* An Anthropic API key, only if you want to use cloud Claude as the brain instead (see [API keys](#api-keys)).

## Install

```bash
# 1. Install Ollama (the local LLM runtime) from https://ollama.com, then pull a model:
ollama pull gemma4:12b                 # the default local brain (~8 GB)

# 2. Install Ada:
cd "AI assistant"                    # the project folder
python3 -m venv .venv
.venv/bin/pip install -e ".[full]"   # full = HUD + menu bar + hotkey + web search extras
.venv/bin/ada setup               # creates ./config.yaml, downloads the whisper model, guides Ollama setup, checks permissions
```

Activate the venv (`source .venv/bin/activate`) or use `.venv/bin/ada` directly. `".[full]"` already includes the HUD; if you only want the on-screen HUD (pywebview) added to a lean install, `pip install -e ".[gui]"`.

Then check everything is healthy:

```bash
ada doctor
```

## API keys

**No keys are required by default** — speech recognition, the brain, and the voice all run locally. Keys only matter if you deliberately switch a stage to a cloud provider. Put any keys in a **`.env` file next to your `config.yaml`** (or export them in your shell). They are read from the environment only — **never put keys in `config.yaml`**, and never commit `.env` to git.

```bash
# .env  (same folder as config.yaml) — all optional
# ANTHROPIC_API_KEY=sk-ant-...      # only if you switch the brain to cloud Claude
                                    # (llm.provider: anthropic). Get one at https://console.anthropic.com
# DEEPGRAM_API_KEY=...              # optional — faster/better cloud speech-to-text
# ELEVENLABS_API_KEY=...            # optional — natural-sounding cloud voice
```

- **`ANTHROPIC_API_KEY`** (optional): only needed if you set `llm.provider: anthropic` to use cloud Claude as the brain instead of the local Ollama model. Not required to run.
- **`DEEPGRAM_API_KEY`** (optional): switches speech-to-text to Deepgram's cloud (set `stt.provider: deepgram`) — faster and more accurate than the local model, but your voice audio is sent to Deepgram.
- **`ELEVENLABS_API_KEY`** (optional): a cloud voice alternative (set `tts.provider: elevenlabs`). You don't need this for a natural voice — the local **Kokoro** voice (`tts.provider: kokoro`, the default in your config) is free and needs no key. Only the reply *text* is ever sent to ElevenLabs, never audio.

## macOS permissions

macOS will prompt you the first time each capability is used. Grant them in **System Settings → Privacy & Security**:

| Permission | Why Ada needs it |
|---|---|
| **Microphone** | Recording your requests (and listening for the wake word if you turn on hands-free). |
| **Input Monitoring** | Required for the push-to-talk hotkey, which is on by default (`hotkey.enabled: true`). Only skippable if you disable the hotkey. |
| **Accessibility** | Moving/resizing/focusing windows via System Events. |
| **Automation (Apple Events)** | Controlling Reminders, Notes, and System Events with AppleScript. You'll get one prompt per target app. |

`ada doctor` reports which permissions look missing.

## Usage

Start the assistant:

```bash
ada run                 # the on-screen HUD window (default) + voice loop
ada run --menubar       # macOS menu-bar icon instead of the HUD
ada run --no-gui        # terminal only (status printed to the console)
```

**The HUD window.** By default `ada run` floats a translucent Jarvis-cyan glass panel over your desktop with the glowing orb at its center. **Tap the orb** (or the talk button, or press **⌘⇧A**) to start a push-to-talk turn; **type in the box** and hit Enter to ask by text; watch your words and Ada's streamed reply appear as chat bubbles. The orb animates with Ada's state — breathing when idle, pulsing while listening, shimmering while thinking, a radial waveform while speaking. Use the header buttons to **mute** or **quit**, and **drag the panel anywhere** (it starts bottom-right and stays above your other windows). It needs no special macOS permission of its own.

Tap **⌘⇧A** (or the orb), wait for the listening chime, then speak. Some things to try:

- "Open Safari." / "Quit Music." *(quitting asks for confirmation)*
- "Find my resume in Documents."
- "Move this window to the left half of the screen."
- "Set the volume to forty percent."
- "Remind me to call the pharmacy at five pm."
- "Make a note: ideas for the demo."
- "Set a timer for ten minutes."
- "What's the tallest building in the world?"

**Interrupting**: just tap **⌘⇧A** again while it is talking — it stops speaking immediately and listens for your new request. (With hands-free wake enabled, saying the wake phrase during playback also interrupts.)

**Mute**: click the mute button in the HUD (or the menu-bar icon → *Mute*). While muted, all microphone input is discarded (the orb/icon shows the muted state).

**Kill switch**: the quit button in the HUD, `Ctrl+C` in the terminal, or menu bar → *Quit*. Any of them stops the mic, the speech, and any in-flight action.

Other commands (all accept `--config PATH`):

```bash
ada setup       # create ./config.yaml, download the whisper model (skipped if STT is cloud), guide Ollama setup when the brain is local, check permissions
ada doctor      # health checks: deps, mic, models, Ollama server + model (or cloud API keys, masked), say/osascript
ada text        # typed REPL for testing without a mic; --speak to also hear replies
ada say "hi"    # test the configured TTS voice
ada listen      # one-shot: record one utterance with VAD and print the transcript
```

## Privacy model

- **Waiting is 100% local.** In the default push-to-talk mode nothing is captured until you tap ⌘⇧A. If you enable hands-free wake, the wake word (openWakeWord) and voice-activity detection (Silero) run on-device — while idle, audio is analyzed in-memory frame by frame and immediately discarded, so nothing is recorded and nothing leaves your Mac while it waits.
- **Transcription is local by default** (faster-whisper). Your voice audio only leaves the machine if you explicitly switch to Deepgram (`stt.provider: deepgram`).
- **Nothing goes to the cloud by default.** With the default local brain (Ollama) plus local STT and TTS, the entire pipeline stays on your Mac. Cloud only happens if you opt in: your request *text* goes to the Anthropic API only if you set `llm.provider: anthropic`; your voice audio goes to Deepgram only if you set `stt.provider: deepgram`; reply text goes to ElevenLabs only if you enable it. Raw room audio is never streamed anywhere.
- **No raw audio is ever stored** — not for the wake word, not for utterances.
- **Audit log**: every tool call (name, arguments, allowed/denied/confirmed, outcome) is appended to `logs/audit.jsonl` next to your `config.yaml`. The app log is `logs/ada.log`. Delete them whenever you like.
- **Visible state**: the menu-bar icon always shows when the mic is live, and chimes mark the start/end of listening.

## Configuration reference

`ada setup` writes a commented `config.yaml`. Config is searched in this order: `--config PATH`, `$ADA_CONFIG`, `./config.yaml`, `~/.config/ada/config.yaml`. The main keys:

| Key | Default | What it does |
|---|---|---|
| `assistant.name` | `Ada` | How the assistant refers to itself. |
| `wake.enabled` | `false` | Hands-free wake-word listening on/off. Off by default — push-to-talk is the default trigger. |
| `wake.model` | `hey_mycroft_v0.1` | Prebuilt openWakeWord phrase, used only when hands-free is on (options: `hey_mycroft` / `alexa` / `hey_rhasspy` / `hey_jarvis`). No prebuilt "hey ada" exists; matching the name needs a custom-trained model. |
| `wake.threshold` | `0.5` | 0–1; higher = fewer false activations (hands-free wake only). |
| `audio.input_device` | `null` | Mic device (null = system default; `ada doctor` lists devices). |
| `vad.end_silence_ms` | `900` | Trailing silence that ends your utterance. |
| `vad.max_utterance_s` | `15.0` | Hard cap on one utterance. |
| `stt.provider` | `local` | `local` (faster-whisper) or `deepgram`. |
| `stt.local.model` | `small.en` | `tiny.en` / `base.en` / `small.en` — bigger = better, slower. |
| `llm.provider` | `ollama` | The brain backend: `ollama` (local & free, needs Ollama running) or `anthropic` (cloud Claude, needs `ANTHROPIC_API_KEY`). |
| `llm.ollama.model` | `gemma4:12b` | Local model tag to run (any tag you've `ollama pull`ed). **Tool calling varies by model** — Gemma chats well but its Ollama tool support is inconsistent; use `qwen2.5` or `llama3.1` for the most reliable Mac actions. |
| `llm.ollama.host` | `http://localhost:11434` | Where the Ollama server is reachable. |
| `llm.anthropic.model` | `claude-opus-4-8` | Cloud Claude model (only used when `llm.provider: anthropic`). |
| `llm.max_tool_iterations` | `6` | Bounded agent loop per request. |
| `tts.provider` | `kokoro` | `kokoro` (natural local neural voice — recommended), `say` (built-in macOS), or `elevenlabs` (cloud). |
| `tts.kokoro.voice` | `af_heart` | Kokoro voice: `af_heart`/`af_bella`/`af_nova`/`af_sarah` (US female), `am_michael`/`am_adam`/`am_puck` (US male), `bf_emma`/`bm_george` (British). |
| `tts.kokoro.speed` | `1.0` | Speaking rate (≈0.8–1.3 sounds natural). |
| `tts.say.voice` | `Samantha` | macOS voice when `tts.provider: say` (`say -v '?'` lists them). |
| `ui.menubar` | `true` | Menu-bar status icon (needs the `menubar` extra). |
| `ui.chimes` | `true` | Audible listening start/stop cues. |
| `hotkey.enabled` | `true` | Push-to-talk hotkey — the default trigger (needs `pynput` + Input Monitoring). |
| `hotkey.combo` | `<cmd>+<shift>+a` | The push-to-talk key combo (⌘⇧A). pynput format, e.g. `<ctrl>+<alt>+a`. |
| `permissions.allowed_app_dirs` | `/Applications`, … | Apps may only be launched from these folders. |
| `permissions.file_roots` | `~/Documents`, `~/Desktop`, `~/Downloads` | File search/open is restricted to these. |
| `permissions.url_schemes` | `http, https` | Only these URL schemes may be opened. |
| `permissions.always_confirm` | `quit_app, system_sleep` | Tools that always show a confirmation dialog. |
| `logging.dir` | `logs` | Log folder (relative to config.yaml). |
| `logging.audit` | `true` | JSONL audit log of every tool call. |

## Troubleshooting

- **⌘⇧A does nothing / no audio input** — check two permissions: **Input Monitoring** (so the push-to-talk hotkey registers) and **Microphone**. System Settings → Privacy & Security → Microphone → enable your terminal app (Terminal/iTerm) or Python; do the same under Input Monitoring. Then restart `ada run`. `ada doctor` will tell you if it cannot read the mic.
- **Model download fails during `ada setup`** — check your network/proxy and disk space, then re-run `ada setup`; downloads resume/retry safely. Whisper models come from Hugging Face — if that is blocked on your network, set `HF_HUB_OFFLINE=0` off VPN or pick a mirror.
- **"… is not allowed assistive access" / Apple Events errors** — grant your terminal app under System Settings → Privacy & Security → **Accessibility** (window control) and **Automation** (Reminders/Notes/System Events). If you previously clicked "Don't Allow", toggle the app off and on again.
- **First transcription is slow** — the whisper model loads lazily on first use (several seconds). Subsequent turns are much faster. `ada run` warms the model at startup; `ada doctor` reports load and transcribe times.
- **Transcription too slow overall** — use a smaller model: set `stt.local.model: base.en` or `tiny.en` in `config.yaml`, or add a `DEEPGRAM_API_KEY` and set `stt.provider: deepgram`.
- **Ada says it can't reach the local model** — the Ollama server isn't running. Start it (open the Ollama app, or run `ollama serve`), make sure the model is pulled (`ollama pull gemma4:12b`), and check `llm.ollama.host` points at where Ollama listens. `ada doctor` reports whether the server is up and the model is present.
- **It chats but won't do actions (open apps, set volume, …)** — your local model doesn't support tool calling through Ollama (Gemma is a common case). Switch `llm.ollama.model` to a tool-capable local model like `qwen2.5` or `llama3.1` (both free and on-device), then restart `ada run`.
- **First reply after starting is slow** — the local model loads into RAM on the first request (cold start); later turns are fast. `keep_alive` (default `10m`) keeps it resident between turns.
- **Wake word too eager / too deaf** (hands-free mode only) — raise or lower `wake.threshold` (try 0.6 for fewer false triggers, 0.4 if it misses you).
- **The HUD window doesn't appear** — it needs pywebview; install it with `pip install -e ".[gui]"` (or `".[full]"`) and restart `ada run`. Without it Ada falls back gracefully to the menu bar (if installed) or the terminal and prints how to enable the HUD. You can also force the other surfaces with `--menubar` or `--no-gui`.
- **No menu-bar icon** (`--menubar` mode) — install the extra (`pip install -e ".[menubar]"`), or use the default HUD, or run with `--no-gui` for the terminal.

## More

Engineering design, architecture diagrams, threading model, and the roadmap live in [`docs/DESIGN.md`](docs/DESIGN.md).
