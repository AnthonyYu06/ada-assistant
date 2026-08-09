"""Tests for ada.brain.knowledge (KnowledgeStore) and ada.tools.memory_tools.

Fully offline and tmp_path-based: the store is exercised against temp JSONL
files, and the tools against a Config whose base_dir is a temp directory.
memory_tools is not in ada.tools._SUBMODULES yet, so tests import it
directly (importlib) after ada.tools.configure(cfg).
"""

from __future__ import annotations

import importlib
import json
import types
from pathlib import Path

import pytest

from ada.brain import knowledge as knowledge_mod
from ada.brain.knowledge import KnowledgeStore


class _FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    """Replace the knowledge module's time binding with a controllable clock."""
    fake = _FakeClock()
    monkeypatch.setattr(knowledge_mod, "time", types.SimpleNamespace(time=fake))
    return fake


# -- KnowledgeStore: add / persistence ----------------------------------------


def test_add_normalizes_whitespace_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "data" / "memory.jsonl"  # parent does not exist yet
    store = KnowledgeStore(path)
    fact_id = store.add("  my   gym\n\tis  Crunch Fitness ")
    assert isinstance(fact_id, str) and fact_id
    assert path.is_file()
    # No stray temp file left behind by the atomic write.
    assert list(path.parent.glob("*.tmp")) == []
    # One JSON object per line with the full record shape.
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["text"] == "my gym is Crunch Fitness"
    assert set(record) == {"id", "text", "created", "last_used", "uses"}


def test_reload_from_disk_preserves_facts(tmp_path: Path) -> None:
    path = tmp_path / "memory.jsonl"
    store = KnowledgeStore(path)
    fid = store.add("prefers Safari over Chrome")
    store.add("coffee order is an oat latte")

    reloaded = KnowledgeStore(path)
    assert len(reloaded) == 2
    hits = reloaded.search("safari")
    assert hits and hits[0]["id"] == fid


def test_add_empty_fact_rejected(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "memory.jsonl")
    with pytest.raises(ValueError):
        store.add("   \n  ")


def test_search_bumps_are_persisted(tmp_path: Path) -> None:
    path = tmp_path / "memory.jsonl"
    store = KnowledgeStore(path)
    store.add("gym is Crunch Fitness on 5th")
    store.search("gym")
    reloaded = KnowledgeStore(path)
    assert reloaded.search("crunch")[0]["uses"] >= 1


# -- KnowledgeStore: dedupe -----------------------------------------------------


def test_dedupe_case_insensitive_exact(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "memory.jsonl")
    first = store.add("Prefers Safari")
    second = store.add("prefers safari")
    assert first == second
    assert len(store) == 1
    assert store.search("safari")[0]["uses"] >= 1  # refreshed, not duplicated


def test_dedupe_containment_keeps_richer_text(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "memory.jsonl")
    first = store.add("gym is Crunch Fitness")
    second = store.add("My gym is Crunch Fitness on 5th")
    assert first == second
    assert len(store) == 1
    assert store.search("gym")[0]["text"] == "My gym is Crunch Fitness on 5th"
    # The shorter, contained restatement also dedupes.
    third = store.add("gym is crunch fitness")
    assert third == first
    assert len(store) == 1


# -- KnowledgeStore: eviction ---------------------------------------------------


def test_eviction_drops_oldest_least_used(tmp_path: Path, clock: _FakeClock) -> None:
    store = KnowledgeStore(tmp_path / "memory.jsonl", max_facts=3)
    store.add("drinks green tea daily")
    clock.advance(10)
    store.add("plays chess on Sundays")
    clock.advance(10)
    store.add("prefers window seats")
    clock.advance(10)
    store.search("green tea")  # bump uses on the oldest fact
    clock.advance(10)
    store.add("birthday is in October")  # overflows max_facts

    texts = {f["text"] for f in store.recent(k=10)}
    assert len(store) == 3
    # "plays chess" was the least-used (0 uses), oldest fact -> evicted.
    assert "plays chess on Sundays" not in texts
    assert texts == {
        "drinks green tea daily",
        "prefers window seats",
        "birthday is in October",
    }


# -- KnowledgeStore: remove -----------------------------------------------------


def test_remove_is_case_insensitive_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "memory.jsonl"
    store = KnowledgeStore(path)
    store.add("coffee order is an oat latte")
    store.add("drinks coffee at 9 every morning")
    store.add("prefers Safari")
    assert store.remove("COFFEE") == 2
    assert store.remove("coffee") == 0  # already gone
    reloaded = KnowledgeStore(path)
    assert len(reloaded) == 1
    assert reloaded.search("safari")


# -- KnowledgeStore: search ranking --------------------------------------------


def test_search_ranks_by_token_overlap(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "memory.jsonl")
    store.add("drinks green tea daily")
    store.add("drinks black coffee before work")
    store.add("prefers Safari")

    hits = store.search("drinks green tea")
    assert [h["text"] for h in hits[:2]] == [
        "drinks green tea daily",          # overlap 3
        "drinks black coffee before work",  # overlap 1
    ]
    assert all("Safari" not in h["text"] for h in hits)  # zero overlap excluded


def test_search_ignores_stopwords(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "memory.jsonl")
    store.add("gym is Crunch Fitness on 5th")
    store.add("prefers the Safari browser")

    hits = store.search("where is the gym")  # only "gym" is a real token
    assert len(hits) == 1
    assert "Crunch" in hits[0]["text"]
    # A query made only of stopwords matches nothing.
    assert store.search("what is the") == []
    assert store.search("") == []


def test_search_ties_break_by_uses(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "memory.jsonl")
    store.add("likes hiking on weekends")
    store.add("likes jazz on weekends")
    store.search("jazz")  # give the jazz fact a use

    hits = store.search("likes weekends")  # equal overlap for both
    assert "jazz" in hits[0]["text"]


def test_recent_orders_by_last_use(tmp_path: Path, clock: _FakeClock) -> None:
    store = KnowledgeStore(tmp_path / "memory.jsonl")
    store.add("drinks green tea daily")
    clock.advance(10)
    store.add("prefers Safari")
    clock.advance(10)
    store.search("green tea")  # refresh the older fact

    recent = store.recent(k=2)
    assert recent[0]["text"] == "drinks green tea daily"
    assert recent[1]["text"] == "prefers Safari"


# -- KnowledgeStore: context_for -------------------------------------------------


def test_context_for_empty_when_no_match(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "memory.jsonl")
    assert store.context_for("anything at all", k=4) == ""
    store.add("prefers Safari")
    assert store.context_for("completely unrelated topic", k=4) == ""


def test_context_for_joins_facts_compactly(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "memory.jsonl")
    store.add("gym is Crunch Fitness on 5th")
    store.add("gym days are Monday and Thursday")

    line = store.context_for("when do I go to the gym", k=4)
    assert line.startswith("Remembered about the user: ")
    assert line.endswith(".")
    assert "; " in line
    assert "Crunch Fitness" in line
    assert "\n" not in line

    single = store.context_for("safari", k=4)
    assert single == ""  # no safari fact yet
    store.add("prefers Safari")
    assert (
        store.context_for("safari", k=4)
        == "Remembered about the user: prefers Safari."
    )


def test_context_for_caps_total_length(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "memory.jsonl")
    for i in range(6):
        store.add(f"project fact number {i} " + f"detail{i} " * 20)
    line = store.context_for("project", k=6)
    assert line != ""
    assert len(line) <= 350
    assert "\n" not in line


# -- KnowledgeStore: corrupt-file tolerance --------------------------------------


def test_corrupt_lines_are_skipped_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "memory.jsonl"
    good = {"id": "abc123", "text": "gym is Crunch", "created": 1.0,
            "last_used": 2.0, "uses": 3}
    path.write_text(
        json.dumps(good) + "\n"
        + "this is not json {{{\n"
        + "42\n"                       # JSON, but not an object
        + '{"text": "   "}\n'          # blank text
        + '{"created": "nope", "text": "x"}\n'  # bad numeric field
        + '{"text": "prefers Safari"}\n'        # sparse but salvageable
        + "\n",
        encoding="utf-8",
    )
    store = KnowledgeStore(path)
    assert len(store) == 2
    hit = store.search("crunch")[0]
    assert hit["id"] == "abc123"
    assert hit["uses"] == 4  # 3 from disk + the search bump
    sparse = store.search("safari")[0]
    assert sparse["id"]  # id was generated for the sparse line


# -- memory tools -----------------------------------------------------------------


@pytest.fixture
def memtools(tmp_path: Path):
    """A Config anchored at tmp_path, wired into ada.tools, plus the module.

    memory_tools is not in load_all()'s _SUBMODULES yet, so it is imported
    directly after configure(cfg), like other standalone tool tests do.
    """
    import ada.tools as tools_pkg
    from ada.config import Config

    cfg = Config()
    cfg.base_dir = tmp_path
    tools_pkg.configure(cfg)
    mod = importlib.import_module("ada.tools.memory_tools")
    return cfg, mod


def test_tools_registered_with_expected_risks(memtools) -> None:
    from ada.tools import registry

    _cfg, _mod = memtools
    assert registry.get("remember").risk == "safe"
    assert registry.get("recall").risk == "safe"
    assert registry.get("usage_patterns").risk == "safe"
    forget_spec = registry.get("forget")
    assert forget_spec.risk == "confirm"
    assert forget_spec.summary({"query": "gym"}) == "Forget everything matching “gym”"


def test_remember_recall_roundtrip(memtools) -> None:
    from ada.tools.registry import ToolResult

    cfg, mod = memtools
    assert mod.remember(fact="My gym is Crunch Fitness on 5th") == "Remembered."
    assert cfg.memory_path.is_file()  # persisted under the config's base_dir

    result = mod.recall(query="gym")
    assert isinstance(result, ToolResult)
    assert result.speech == "Here's what I remember."
    assert result.display["kind"] == "list"
    assert result.display["title"] == "Memory"
    items = result.display["items"]
    assert len(items) == 1
    assert "Crunch Fitness" in items[0]["text"]
    assert items[0]["sub"] == "just now"


def test_recall_empty_query_lists_recent(memtools) -> None:
    from ada.tools.registry import ToolResult

    _cfg, mod = memtools
    mod.remember(fact="prefers Safari")
    mod.remember(fact="coffee order is an oat latte")
    result = mod.recall()
    assert isinstance(result, ToolResult)
    assert len(result.display["items"]) == 2


def test_recall_no_match_is_plain_speech(memtools) -> None:
    _cfg, mod = memtools
    mod.remember(fact="prefers Safari")
    assert mod.recall(query="zebra migration") == "I don't have anything about that."


def test_forget_roundtrip_and_no_match_error(memtools) -> None:
    from ada.tools.registry import ToolError

    _cfg, mod = memtools
    mod.remember(fact="gym is Crunch Fitness on 5th")
    mod.remember(fact="gym days are Monday and Thursday")
    assert mod.forget(query="gym") == "Forgot 2 things."
    assert mod.recall(query="gym") == "I don't have anything about that."
    with pytest.raises(ToolError):
        mod.forget(query="gym")


def test_disabled_memory_raises_on_every_tool(memtools) -> None:
    from ada.tools.registry import ToolError

    cfg, mod = memtools
    cfg.memory.enabled = False
    expected = "Long-term memory is turned off in the config."
    for call in (
        lambda: mod.remember(fact="x"),
        lambda: mod.forget(query="x"),
        lambda: mod.recall(query="x"),
        lambda: mod.usage_patterns(),
    ):
        with pytest.raises(ToolError, match="turned off"):
            call()
    with pytest.raises(ToolError) as excinfo:
        mod.remember(fact="x")
    assert str(excinfo.value) == expected


def test_get_store_is_shared_and_stable(memtools) -> None:
    _cfg, mod = memtools
    assert mod.get_store() is mod.get_store()


# -- usage_patterns ----------------------------------------------------------------


def _audit_line(kind: str, hour: int, **fields) -> str:
    record = {"ts": f"2026-07-10T{hour:02d}:15:00+00:00", "kind": kind, **fields}
    return json.dumps(record)


def _write_audit(cfg, lines: list[str]) -> Path:
    path = cfg.log_dir / "audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_usage_patterns_without_history(memtools) -> None:
    _cfg, mod = memtools
    assert mod.usage_patterns() == "I don't have enough history yet."


def test_usage_patterns_empty_audit_file(memtools) -> None:
    cfg, mod = memtools
    _write_audit(cfg, [""])
    assert mod.usage_patterns() == "I don't have enough history yet."


def test_usage_patterns_chart_when_hours_spread(memtools) -> None:
    from ada.tools.registry import ToolResult

    cfg, mod = memtools
    lines = []
    # open_app across six distinct hours (>=5 -> chart), Safari dominant.
    for hour, app in zip(
        (8, 9, 10, 11, 12, 13), ("Safari", "Safari", "Safari", "Safari", "Mail", "Mail")
    ):
        lines.append(_audit_line(
            "tool_call", hour, tool="open_app",
            args={"app_name": app}, allowed=True, ok=True,
        ))
    lines.append(_audit_line("tool_call", 9, tool="search_web",
                             args={"query": "weather"}, allowed=True, ok=True))
    lines.append(_audit_line("transcript", 9, text="open safari"))
    lines.append("not json at all {{{")  # corrupt audit lines are tolerated
    _write_audit(cfg, lines)

    result = mod.usage_patterns()
    assert isinstance(result, ToolResult)
    assert "open apps" in result.speech
    assert "Safari" in result.speech
    assert "usually around" in result.speech
    chart = result.display
    assert chart["kind"] == "chart"
    assert chart["chart"] == "bar"
    assert chart["title"] == "When you talk to Ada"
    assert len(chart["labels"]) == len(chart["values"]) >= 5
    assert sum(chart["values"]) == 8  # 7 tool_calls + 1 transcript


def test_usage_patterns_keyvalue_when_few_hours(memtools) -> None:
    from ada.tools.registry import ToolResult

    cfg, mod = memtools
    lines = [
        _audit_line("tool_call", 9, tool="open_app",
                    args={"app_name": "Safari"}, allowed=True, ok=True)
        for _ in range(3)
    ]
    lines.append(_audit_line("transcript", 9, text="open safari"))
    _write_audit(cfg, lines)

    result = mod.usage_patterns()
    assert isinstance(result, ToolResult)
    display = result.display
    assert display["kind"] == "keyvalue"
    pairs = dict(tuple(pair) for pair in display["pairs"])
    assert pairs["open_app"] == "3"
    assert pairs["Most-opened app"] == "Safari"
    assert pairs["Voice commands"] == "1"
    assert "Busiest hour" in pairs
