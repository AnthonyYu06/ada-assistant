"""Tests for the local Ollama brain and the provider factory.

Everything is mocked — no Ollama server, no network, no tools that mutate the
system (the one tool exercised, set_volume, is monkeypatched to a no-op).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest


@pytest.fixture
def loaded(config):
    """A Config with the tool registry populated (so tool calls resolve)."""
    from ada import tools

    tools.load_all(config)
    return config


class _FakeStreamResponse:
    def __init__(self, lines: list[str], status: int = 200, error_body: bytes = b"") -> None:
        self._lines = lines
        self.status_code = status
        self._error_body = error_body

    def __enter__(self) -> "_FakeStreamResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._error_body

    def iter_lines(self):
        yield from self._lines


class _FakeHTTPResponse:
    """A non-streaming httpx response (warmup's GET /api/tags + POST /api/chat)."""

    def __init__(self, status: int = 200, payload: Any = None, content: bytes = b"") -> None:
        self.status_code = status
        self._payload = payload
        self.content = content

    def json(self) -> Any:
        return self._payload


class _FakeClient:
    """Stands in for the OllamaBrain's httpx.Client."""

    def __init__(
        self,
        responses: list[Any],
        post_responses: list[Any] | None = None,
        tags: tuple[str, ...] = ("gemma4:12b",),
    ) -> None:
        # Each entry is either a fake response or an Exception to raise.
        self._responses = list(responses)
        self._post_responses = list(post_responses or [])
        self._tags = list(tags)
        self.bodies: list[dict] = []
        self.post_bodies: list[dict] = []
        self.closed = False

    def stream(self, method: str, url: str, json: dict | None = None):
        self.bodies.append(json or {})
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    def get(self, url: str, timeout: float | None = None):
        return _FakeHTTPResponse(200, {"models": [{"name": n} for n in self._tags]})

    def post(self, url: str, json: dict | None = None, timeout: float | None = None):
        self.post_bodies.append(json or {})
        nxt = self._post_responses.pop(0) if self._post_responses else _FakeHTTPResponse(200, {})
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    def close(self) -> None:
        self.closed = True


def _line(**message: Any) -> str:
    done = message.pop("done", False)
    return json.dumps({"message": message, "done": done})


def _make_brain(cfg, client):
    from ada.brain.ollama_client import OllamaBrain
    from ada.safety import AuditLog
    from ada.safety.confirm import ConsoleConfirmer
    from ada.safety.permissions import Policy

    audit = AuditLog(cfg.log_dir / "audit_test.jsonl", enabled=False)
    brain = OllamaBrain(cfg, Policy(cfg), ConsoleConfirmer(), audit)
    brain._client = client  # replace the real httpx client
    return brain


def test_tool_loop_runs_and_streams(loaded, monkeypatch):
    from ada.tools import registry

    monkeypatch.setattr(
        registry.get("set_volume"), "handler", lambda level: f"Volume set to {level} percent."
    )

    # Turn 1: greeting text + a tool call. Turn 2: the final spoken answer.
    turn1 = _FakeStreamResponse([
        _line(role="assistant", content="Sure. "),
        _line(role="assistant", content="",
              tool_calls=[{"function": {"name": "set_volume", "arguments": {"level": 40}}}]),
        _line(role="assistant", content="", done=True),
    ])
    turn2 = _FakeStreamResponse([
        _line(role="assistant", content="Volume set to 40 percent.", done=True),
    ])
    client = _FakeClient([turn1, turn2])
    brain = _make_brain(loaded, client)

    spoken: list[str] = []
    out = brain.handle("set the volume to forty", on_sentence=spoken.append)

    assert "Sure." in " ".join(spoken)
    assert "Volume set to 40 percent." in out
    # The first request carried the tools; a tool-role result is in history.
    assert "tools" in client.bodies[0]
    tool_msgs = [m for m in brain.memory.messages if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_name"] == "set_volume"
    assert "40" in tool_msgs[0]["content"]


def test_tools_unsupported_falls_back_to_chat(loaded):
    # First call: 400 "does not support tools". Retry (no tools): a plain reply.
    unsupported = _FakeStreamResponse(
        [], status=400, error_body=b'{"error":"gemma3:4b does not support tools"}'
    )
    retry = _FakeStreamResponse([_line(role="assistant", content="Hello there.", done=True)])
    client = _FakeClient([unsupported, retry])
    brain = _make_brain(loaded, client)

    spoken: list[str] = []
    out = brain.handle("hi", on_sentence=spoken.append)

    assert brain._tools_supported is False
    assert any("can't run Mac actions" in s for s in spoken)
    assert "Hello there." in out
    # First body had tools; the retry body did not.
    assert "tools" in client.bodies[0]
    assert "tools" not in client.bodies[1]


def test_server_unreachable_is_spoken(loaded):
    client = _FakeClient([httpx.ConnectError("connection refused")])
    brain = _make_brain(loaded, client)

    spoken: list[str] = []
    out = brain.handle("hi", on_sentence=spoken.append)

    assert "can't reach the local model" in out.lower()
    assert spoken and "can't reach the local model" in spoken[0].lower()


def test_factory_selects_provider(loaded):
    from ada.brain import AnthropicBrain, OllamaBrain, create_brain
    from ada.safety import AuditLog
    from ada.safety.confirm import ConsoleConfirmer
    from ada.safety.permissions import Policy

    audit = AuditLog(loaded.log_dir / "a.jsonl", enabled=False)
    policy, confirmer = Policy(loaded), ConsoleConfirmer()

    loaded.llm.provider = "ollama"
    b = create_brain(loaded, policy, confirmer, audit)
    assert isinstance(b, OllamaBrain)
    b.close()

    # anthropic provider with no key must fail loudly (not silently no-op).
    loaded.llm.provider = "anthropic"
    loaded.anthropic_api_key = None
    with pytest.raises(RuntimeError):
        create_brain(loaded, policy, confirmer, audit)

    loaded.llm.provider = "nonsense"
    with pytest.raises(ValueError):
        create_brain(loaded, policy, confirmer, audit)


def test_coerce_args_handles_string_json():
    from ada.brain.ollama_client import OllamaBrain

    assert OllamaBrain._coerce_args({"a": 1}) == {"a": 1}
    assert OllamaBrain._coerce_args('{"a": 1}') == {"a": 1}
    assert OllamaBrain._coerce_args("not json") == {}
    assert OllamaBrain._coerce_args(None) == {}


# -- thinking control ---------------------------------------------------------


def test_think_param_follows_config(loaded):
    # Default (think: false) — the parameter is sent explicitly as False.
    client = _FakeClient([_FakeStreamResponse([_line(role="assistant", content="Hi.", done=True)])])
    brain = _make_brain(loaded, client)
    brain.handle("hi", on_sentence=lambda s: None)
    assert client.bodies[0]["think"] is False

    # Flipping the config flag flips the request parameter.
    loaded.llm.ollama.think = True
    client = _FakeClient([_FakeStreamResponse([_line(role="assistant", content="Hi.", done=True)])])
    brain = _make_brain(loaded, client)
    brain.handle("hi", on_sentence=lambda s: None)
    assert client.bodies[0]["think"] is True


@pytest.mark.parametrize(
    "error_body",
    [
        b'{"error":"\\"gemma4:12b\\" does not support thinking"}',   # model capability
        b'{"error":"json: unknown field \\"think\\""}',              # pre-think server
    ],
)
def test_think_rejected_retries_without_and_remembers(loaded, error_body):
    rejected = _FakeStreamResponse([], status=400, error_body=error_body)
    retry = _FakeStreamResponse([_line(role="assistant", content="Hello there.", done=True)])
    later = _FakeStreamResponse([_line(role="assistant", content="Still here.", done=True)])
    client = _FakeClient([rejected, retry, later])
    brain = _make_brain(loaded, client)

    spoken: list[str] = []
    out = brain.handle("hi", on_sentence=spoken.append)

    assert brain._think_supported is False
    assert "Hello there." in out
    # Retried once without `think`; tools were untouched by the fallback.
    assert "think" in client.bodies[0]
    assert "think" not in client.bodies[1]
    assert "tools" in client.bodies[1]
    # No error/heads-up sentence was spoken — the fallback is silent.
    assert all("error" not in s.lower() for s in spoken)

    # Subsequent requests skip the parameter without another round trip.
    brain.handle("again", on_sentence=spoken.append)
    assert len(client.bodies) == 3
    assert "think" not in client.bodies[2]


def test_thinking_deltas_are_never_spoken_or_stored(loaded):
    stream = _FakeStreamResponse([
        _line(role="assistant", thinking="The user greeted me. "),
        _line(role="assistant", thinking="I should respond warmly.", content=""),
        _line(role="assistant", content="Hello!"),
        _line(role="assistant", content="", done=True),
    ])
    client = _FakeClient([stream])
    brain = _make_brain(loaded, client)

    spoken: list[str] = []
    out = brain.handle("hi", on_sentence=spoken.append)

    assert "Hello!" in out
    assert any("Hello!" in s for s in spoken)
    # The reasoning text never reaches the splitter, the answer, or memory.
    assert "greeted" not in out and all("greeted" not in s for s in spoken)
    stored = [m for m in brain.memory.messages if m.get("role") == "assistant"]
    assert stored and stored[0]["content"] == "Hello!"


# -- warmup -------------------------------------------------------------------


def test_warmup_primes_the_prompt_cache(loaded):
    client = _FakeClient([], post_responses=[_FakeHTTPResponse(200, {})])
    brain = _make_brain(loaded, client)
    brain.warmup()

    assert len(client.post_bodies) == 1
    body = client.post_bodies[0]
    assert body["model"] == loaded.llm.ollama.model
    assert body["messages"] == [brain._system_msg]
    assert body["tools"] == brain._tools
    assert body["think"] is False
    assert body["stream"] is False
    assert body["keep_alive"] == loaded.llm.ollama.keep_alive
    assert body["options"]["temperature"] == loaded.llm.ollama.temperature
    assert body["options"]["num_ctx"] == loaded.llm.ollama.num_ctx
    assert body["options"]["num_predict"] == 1

    # A real turn must match the warmed prefix exactly (minus num_predict).
    client._responses.append(
        _FakeStreamResponse([_line(role="assistant", content="Hi.", done=True)])
    )
    brain.handle("hi", on_sentence=lambda s: None)
    turn = client.bodies[0]
    warm_options = {k: v for k, v in body["options"].items() if k != "num_predict"}
    assert turn["options"] == warm_options
    assert turn["keep_alive"] == body["keep_alive"]
    assert turn["think"] == body["think"]
    assert turn["tools"] == body["tools"]
    assert turn["messages"][0] == brain._system_msg


def test_warmup_think_rejection_disables_think(loaded):
    rejected = _FakeHTTPResponse(
        400, content=b'{"error":"\\"gemma4:12b\\" does not support thinking"}'
    )
    client = _FakeClient([], post_responses=[rejected, _FakeHTTPResponse(200, {})])
    brain = _make_brain(loaded, client)
    brain.warmup()

    assert brain._think_supported is False
    assert len(client.post_bodies) == 2
    assert "think" in client.post_bodies[0]
    assert "think" not in client.post_bodies[1]

    # Real turns afterwards also omit the parameter.
    client._responses.append(
        _FakeStreamResponse([_line(role="assistant", content="Hi.", done=True)])
    )
    brain.handle("hi", on_sentence=lambda s: None)
    assert "think" not in client.bodies[0]


def test_warmup_stays_nonfatal(loaded):
    # Server down: warmup returns quietly, no preload is attempted.
    class _DownClient(_FakeClient):
        def get(self, url: str, timeout: float | None = None):
            raise httpx.ConnectError("connection refused")

    down = _DownClient([])
    _make_brain(loaded, down).warmup()
    assert down.post_bodies == []

    # Model not pulled: the presence check stops before the preload POST.
    absent = _FakeClient([], tags=("qwen2.5:7b",))
    _make_brain(loaded, absent).warmup()
    assert absent.post_bodies == []

    # Preload POST itself failing is swallowed too.
    flaky = _FakeClient([], post_responses=[httpx.ConnectTimeout("timed out")])
    brain = _make_brain(loaded, flaky)
    brain.warmup()
    assert brain._think_supported is True  # an outage is not a rejection
