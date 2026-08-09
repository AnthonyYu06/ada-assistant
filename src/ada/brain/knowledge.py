"""Long-term knowledge: a small persistent fact store.

Facts the user asks Ada to remember ("my gym is Crunch Fitness on 5th")
live in one plain JSONL file on disk — one JSON object per line with
``{id, text, created, last_used, uses}``. That format is the privacy
stance: everything Ada knows about the user sits in a single local,
human-readable file the user can open, grep, edit, or delete at any
time. There are no embeddings, no vector database, and nothing ever
leaves the machine.

Retrieval is deliberately simple — lowercase token overlap with a tiny
stopword list — because a few hundred short facts don't need anything
smarter, and simple ranking is inspectable and predictable.

``KnowledgeStore`` is thread-safe (tool handlers run on worker threads)
and persists atomically (write a temp file, then ``os.replace``), so a
crash mid-save never corrupts the memory file.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger("ada.brain.knowledge")

# Joined-fact context line length cap (it is prefixed into the model's
# user message, so it must stay compact).
_CONTEXT_MAX_CHARS = 350

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Tiny stopword set: only glue words that carry no retrieval signal.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "been",
    "i", "me", "my", "we", "our", "you", "your", "it", "its", "s",
    "of", "to", "in", "on", "at", "for", "with", "and", "or", "not",
    "that", "this", "these", "those", "what", "which", "who", "when",
    "where", "how", "why", "do", "does", "did", "have", "has", "had",
    "about", "please", "user",
})


def _tokenize(text: str) -> set[str]:
    """Lowercased alphanumeric tokens with stopwords removed."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _normalize(text: str) -> str:
    """Collapse all whitespace (including newlines) to single spaces."""
    return " ".join(str(text).split())


class KnowledgeStore:
    """Thread-safe persistent fact store backed by one JSONL file.

    Loading is lazy (first access), the file and its parent directory may
    not exist yet, and corrupt lines are skipped with a warning — a damaged
    memory file degrades, it never takes Ada down.
    """

    def __init__(self, path: Path, max_facts: int = 400) -> None:
        self.path = Path(path)
        self.max_facts = max_facts
        self._lock = threading.RLock()
        self._facts: list[dict[str, Any]] | None = None  # lazy

    # -- persistence ---------------------------------------------------------

    def _ensure_loaded(self) -> list[dict[str, Any]]:
        """Load facts from disk once. Must be called with the lock held."""
        if self._facts is not None:
            return self._facts
        facts: list[dict[str, Any]] = []
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raw = ""
        except OSError as exc:
            log.warning("Could not read memory file %s: %s", self.path, exc)
            raw = ""
        for lineno, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            fact = self._parse_line(line)
            if fact is None:
                log.warning(
                    "Skipping corrupt memory line %d in %s", lineno, self.path
                )
                continue
            facts.append(fact)
        self._facts = facts
        return facts

    @staticmethod
    def _parse_line(line: str) -> dict[str, Any] | None:
        """One JSONL line -> a well-formed fact dict, or None if corrupt."""
        try:
            obj = json.loads(line)
        except ValueError:
            return None
        if not isinstance(obj, dict):
            return None
        text = obj.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        now = time.time()
        try:
            created = float(obj.get("created", now))
            last_used = float(obj.get("last_used", created))
            uses = int(obj.get("uses", 0))
        except (TypeError, ValueError):
            return None
        fact_id = obj.get("id")
        if not isinstance(fact_id, str) or not fact_id:
            fact_id = uuid.uuid4().hex[:12]
        return {
            "id": fact_id,
            "text": _normalize(text),
            "created": created,
            "last_used": last_used,
            "uses": uses,
        }

    def _save(self) -> None:
        """Atomically persist all facts. Must be called with the lock held."""
        facts = self._facts or []
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for fact in facts:
                    fh.write(json.dumps(fact, ensure_ascii=False) + "\n")
            os.replace(tmp, self.path)
        except OSError:
            log.exception("Could not persist memory to %s", self.path)

    # -- public API ----------------------------------------------------------

    def add(self, text: str) -> str:
        """Store a fact; dedupe against near-identical ones. Returns its id.

        Duplicate detection is case-insensitive: an exact match, or one text
        containing the other, refreshes the existing fact (keeping the longer
        wording) instead of creating a near-copy.
        """
        norm = _normalize(text)
        if not norm:
            raise ValueError("Cannot remember an empty fact.")
        now = time.time()
        with self._lock:
            facts = self._ensure_loaded()
            low = norm.lower()
            for fact in facts:
                existing = fact["text"].lower()
                if existing == low or low in existing or existing in low:
                    if len(norm) > len(fact["text"]):
                        fact["text"] = norm  # keep the richer wording
                    fact["last_used"] = now
                    fact["uses"] = int(fact["uses"]) + 1
                    self._save()
                    return fact["id"]
            fact = {
                "id": uuid.uuid4().hex[:12],
                "text": norm,
                "created": now,
                "last_used": now,
                "uses": 0,
            }
            facts.append(fact)
            while len(facts) > self.max_facts:
                # Evict the least-used fact, oldest first among ties. The
                # fact just added has the newest last_used, so it survives.
                victim = min(
                    facts, key=lambda f: (f["uses"], f["last_used"], f["created"])
                )
                facts.remove(victim)
            self._save()
            return fact["id"]

    def remove(self, query: str) -> int:
        """Delete facts whose text contains `query` (case-insensitive)."""
        needle = _normalize(query).lower()
        if not needle:
            return 0
        with self._lock:
            facts = self._ensure_loaded()
            kept = [f for f in facts if needle not in f["text"].lower()]
            removed = len(facts) - len(kept)
            if removed:
                self._facts = kept
                self._save()
            return removed

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Top-k facts ranked by token overlap with `query`.

        Ties break by uses, then recency. Only facts sharing at least one
        (non-stopword) token are returned. Returned facts get their
        last_used/uses refreshed.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        with self._lock:
            facts = self._ensure_loaded()
            scored = []
            for fact in facts:
                overlap = len(query_tokens & _tokenize(fact["text"]))
                if overlap >= 1:
                    scored.append((overlap, fact["uses"], fact["last_used"], fact))
            scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
            top = [fact for _, _, _, fact in scored[: max(0, k)]]
            if top:
                now = time.time()
                for fact in top:
                    fact["last_used"] = now
                    fact["uses"] = int(fact["uses"]) + 1
                self._save()
            return [dict(fact) for fact in top]

    def recent(self, k: int = 10) -> list[dict[str, Any]]:
        """The k most recently used/added facts, newest first."""
        with self._lock:
            facts = self._ensure_loaded()
            ordered = sorted(
                facts, key=lambda f: (f["last_used"], f["created"]), reverse=True
            )
            return [dict(fact) for fact in ordered[: max(0, k)]]

    def context_for(self, text: str, k: int = 4) -> str:
        """A single compact context line of remembered facts relevant to `text`.

        Returns "" when nothing matches. Otherwise something like
        ``Remembered about the user: gym is Crunch Fitness on 5th; prefers
        Safari.`` — one line, no newlines, capped at ~350 characters, meant
        to be prefixed into the model's user message.
        """
        matches = self.search(text, k=k)
        if not matches:
            return ""
        prefix = "Remembered about the user: "
        parts: list[str] = []
        for fact in matches:
            part = fact["text"].strip().rstrip(".")
            if not part:
                continue
            candidate = prefix + "; ".join([*parts, part]) + "."
            if parts and len(candidate) > _CONTEXT_MAX_CHARS:
                break
            parts.append(part)
        if not parts:
            return ""
        line = prefix + "; ".join(parts) + "."
        if len(line) > _CONTEXT_MAX_CHARS:
            line = line[: _CONTEXT_MAX_CHARS - 1].rstrip() + "…"
        return line

    def __len__(self) -> int:
        with self._lock:
            return len(self._ensure_loaded())
