"""Incremental sentence splitting for streaming TTS.

Text deltas from the model stream are accumulated and cut into chunks that
are natural to speak: complete sentences (or small groups of very short
ones), so the voice can start talking before the full response has arrived.
The very first chunk of a stream may additionally be cut early at a clause
boundary (see _early_first_chunk) to shave seconds off the first word.
"""

from __future__ import annotations

# Quote / bracket characters that may trail sentence-ending punctuation
# ("He said 'stop.'" / "(Really?)").
_CLOSERS = "\"'’”)]"

# Abbreviations whose trailing period does not end a sentence. Compared
# against the lowercased token immediately before the period.
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "st", "prof", "sr", "jr",
    "e.g", "i.e", "etc", "vs",
})

# First-emission latency cut: only a stream's OPENING chunk may be released
# early — at a clause boundary once _FIRST_CLAUSE_WORDS words have buffered,
# or unconditionally once _FIRST_FORCE_WORDS complete words have arrived with
# no usable boundary. Everything after the first emission keeps full-sentence
# behavior (synthesis of later sentences overlaps playback, so only the first
# chunk is on the critical path to Ada's first audible word).
_FIRST_CLAUSE_WORDS = 6
_FIRST_FORCE_WORDS = 12
_CLAUSE_MARKS = ",;—"  # comma, semicolon, em-dash


class SentenceSplitter:
    """Accumulate streamed text and emit speakable chunks."""

    def __init__(self, min_chunk: int = 15, max_buffer: int = 250) -> None:
        self.min_chunk = min_chunk
        self.max_buffer = max_buffer
        self._buf = ""
        # True once this stream has emitted its first chunk; until then feed()
        # may cut an early clause-sized chunk so the voice starts sooner.
        self._first_done = False

    def feed(self, delta: str) -> list[str]:
        """Add a streamed delta; return any chunks now ready to speak."""
        self._buf += delta
        out: list[str] = []
        while True:
            chunk = self._next_chunk()
            if chunk is None:
                break
            if chunk:
                out.append(chunk)
        # No full sentence ready and nothing spoken yet this stream: try an
        # early first chunk (clause boundary / forced word cap) so playback
        # can begin before the first sentence finishes arriving.
        if not out and not self._first_done:
            early = self._early_first_chunk()
            if early:
                out.append(early)
        if out:
            self._first_done = True
        return out

    def flush(self) -> str:
        """Return and clear whatever remains in the buffer.

        Ends the stream: the next feed() starts a new one and is again
        eligible for an early first chunk.
        """
        chunk = self._buf.strip()
        self._buf = ""
        self._first_done = False
        return chunk

    # -- internals -----------------------------------------------------------
    def _early_first_chunk(self) -> str | None:
        """Cut the stream's OPENING chunk early, or return None if not ready.

        Synthesis runs at ~0.5x realtime, so waiting for the first full
        sentence delays Ada's first audible word by seconds. Until the first
        emission we therefore split:
        - at the first clause mark (comma / semicolon / em-dash) preceded by
          at least _FIRST_CLAUSE_WORDS words — after the mark, punctuation
          kept, so the chunk still reads naturally with a clause pause;
        - failing that, once _FIRST_FORCE_WORDS complete words have buffered —
          at the last whitespace, so a token still arriving mid-delta is
          never split in half.
        A comma/semicolon must be followed by whitespace to count ("3,500" is
        not a clause; at end-of-buffer we wait for the next delta to decide,
        like _next_chunk does for sentence punctuation). An em-dash needs no
        following whitespace ("today—perfect").
        """
        buf = self._buf
        n = len(buf)
        for i, ch in enumerate(buf):
            if ch not in _CLAUSE_MARKS:
                continue
            if ch != "—" and (i + 1 >= n or not buf[i + 1].isspace()):
                continue
            if len(buf[:i].split()) < _FIRST_CLAUSE_WORDS:
                continue
            self._buf = buf[i + 1 :]
            return buf[: i + 1].strip()

        # Words already fully delivered: the trailing token only counts if the
        # buffer ends in whitespace (otherwise it may still be streaming in).
        words = buf.split()
        complete = len(words) if buf[-1:].isspace() else len(words) - 1
        if complete >= _FIRST_FORCE_WORDS:
            j = n
            while j > 0 and not buf[j - 1].isspace():
                j -= 1
            if j > 0:
                chunk = buf[:j].strip()
                self._buf = buf[j:]
                return chunk
        return None

    def _next_chunk(self) -> str | None:
        """Extract the next chunk from the buffer, or None if none is ready.

        May return an empty string (e.g. a leading blank line); callers skip
        those but keep scanning.
        """
        buf = self._buf
        n = len(buf)
        i = 0
        while i < n:
            ch = buf[i]
            if ch == "\n":
                j = i
                while j < n and buf[j] == "\n":
                    j += 1
                forced = (j - i) >= 2  # blank line: flush even a short chunk
                chunk = buf[:i].strip()
                if forced or len(chunk) >= self.min_chunk:
                    self._buf = buf[j:]
                    return chunk
                i = j
                continue
            if ch in ".!?":
                j = i + 1
                while j < n and buf[j] in _CLOSERS:
                    j += 1
                if j >= n:
                    # Punctuation at the very end — we can't yet tell whether
                    # whitespace follows. Wait for more input.
                    break
                if not buf[j].isspace():
                    i += 1
                    continue
                if ch == "." and _non_terminal_period(buf, i):
                    i += 1
                    continue
                chunk = buf[:j].strip()
                if len(chunk) >= self.min_chunk:
                    self._buf = buf[j:]
                    return chunk
                i = j
                continue
            i += 1

        # No usable sentence boundary. Guard against unbounded growth: emit
        # up to the last space (within the cap) so TTS is never starved by
        # run-on output.
        if len(self._buf) > self.max_buffer:
            cut = self._buf.rfind(" ", 0, self.max_buffer + 1)
            if cut <= 0:
                cut = self._buf.find(" ")  # pathological run-on token
            if cut <= 0:
                chunk = self._buf.strip()
                self._buf = ""
            else:
                chunk = self._buf[:cut].strip()
                self._buf = self._buf[cut + 1 :]
            return chunk
        return None


def _non_terminal_period(buf: str, i: int) -> bool:
    """True if the period at buf[i] does not end a sentence."""
    # Decimal number: "3.5".
    if 0 < i < len(buf) - 1 and buf[i - 1].isdigit() and buf[i + 1].isdigit():
        return True
    # Token immediately before the period (letters and internal periods,
    # so "e.g." yields "e.g" at its final period).
    k = i - 1
    while k >= 0 and (buf[k].isalpha() or buf[k] == "."):
        k -= 1
    token = buf[k + 1 : i].strip(".")
    if not token:
        return False
    if token.lower() in _ABBREVIATIONS:
        return True
    # Single initials ("J. Smith") and dotted acronyms ("U.S.", "a.m.").
    parts = [p for p in token.split(".") if p]
    if parts and all(len(p) == 1 for p in parts):
        return True
    return False
