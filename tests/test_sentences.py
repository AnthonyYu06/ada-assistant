"""Tests for ada.brain.sentences.SentenceSplitter: streaming TTS chunking."""

from __future__ import annotations

from ada.brain.sentences import SentenceSplitter


def _feed_all(splitter: SentenceSplitter, deltas: list[str]) -> list[str]:
    chunks: list[str] = []
    for delta in deltas:
        chunks.extend(splitter.feed(delta))
    return chunks


def test_streaming_deltas_produce_sentences() -> None:
    s = SentenceSplitter(min_chunk=1)
    chunks = _feed_all(
        s,
        ["The weather", " is sunny today", ". Tomorrow", " looks cloudy."],
    )
    assert chunks == ["The weather is sunny today."]
    assert s.flush() == "Tomorrow looks cloudy."


def test_trailing_punctuation_waits_for_more_input() -> None:
    # A '.' at the very end of the buffer is ambiguous ("3." might continue
    # as "3.5") — nothing is emitted until the following character arrives.
    s = SentenceSplitter(min_chunk=1)
    assert s.feed("This one is done.") == []
    assert s.feed(" Next") == ["This one is done."]


def test_short_sentences_group_until_min_chunk() -> None:
    s = SentenceSplitter(min_chunk=15)
    chunks = _feed_all(s, ["Hi. ", "There. ", "All good now, friends. "])
    assert chunks == ["Hi. There. All good now, friends."]


def test_question_and_exclamation_boundaries() -> None:
    s = SentenceSplitter(min_chunk=1)
    chunks = _feed_all(s, ["Are you sure? Absolutely certain! Then go ahead."])
    assert chunks == ["Are you sure?", "Absolutely certain!"]
    assert s.flush() == "Then go ahead."


def test_closing_quote_included_in_chunk() -> None:
    s = SentenceSplitter(min_chunk=1)
    chunks = _feed_all(s, ['He said "stop." Then he left. '])
    assert chunks == ['He said "stop."', "Then he left."]


def test_abbreviations_not_split() -> None:
    s = SentenceSplitter(min_chunk=1)
    chunks = _feed_all(s, ["Dr. Smith saw Mr. Jones at the corner. Then he left. "])
    assert chunks == [
        "Dr. Smith saw Mr. Jones at the corner.",
        "Then he left.",
    ]


def test_eg_abbreviation_not_split() -> None:
    s = SentenceSplitter(min_chunk=1)
    chunks = _feed_all(s, ["Bring fruit, e.g. apples and pears. Done deal. "])
    assert chunks == ["Bring fruit, e.g. apples and pears.", "Done deal."]


def test_initials_and_acronyms_not_split() -> None:
    s = SentenceSplitter(min_chunk=1)
    chunks = _feed_all(s, ["The U.S. team met J. Smith at noon. Great outcome. "])
    assert chunks == ["The U.S. team met J. Smith at noon.", "Great outcome."]


def test_decimals_not_split() -> None:
    s = SentenceSplitter(min_chunk=1)
    chunks = _feed_all(s, ["Pi is about 3.14159 in most cases. Neat fact. "])
    assert chunks == ["Pi is about 3.14159 in most cases.", "Neat fact."]


def test_blank_line_forces_flush_of_short_chunk() -> None:
    s = SentenceSplitter(min_chunk=15)
    chunks = _feed_all(s, ["Okay.\n\nSecond paragraph is here.\n"])
    assert chunks == ["Okay.", "Second paragraph is here."]


def test_flush_returns_remainder_and_clears() -> None:
    s = SentenceSplitter(min_chunk=1)
    assert s.feed("and then some trailing words") == []
    assert s.flush() == "and then some trailing words"
    assert s.flush() == ""  # buffer was cleared
    assert s.feed("Fresh start here. Next") == ["Fresh start here."]


def test_long_buffer_force_emits_at_a_space() -> None:
    s = SentenceSplitter(min_chunk=5, max_buffer=40)
    text = ("alpha bravo charlie delta " * 4).strip()  # no sentence punctuation
    chunks = s.feed(text)
    assert chunks, "run-on text past max_buffer must still emit"
    for chunk in chunks:
        assert chunk
        assert len(chunk) <= 40
        assert not chunk.endswith(" ")
    remainder = s.flush()
    # Cuts happened exactly at spaces: the pieces reassemble the input.
    assert " ".join([*chunks, remainder]).strip() == text


def test_pathological_runon_token_still_emits() -> None:
    s = SentenceSplitter(min_chunk=5, max_buffer=50)
    token = "x" * 300  # a single 300-char "word", no spaces at all
    chunks = s.feed(token)
    assert chunks == [token]
    assert s.flush() == ""


def test_empty_feed_and_flush() -> None:
    s = SentenceSplitter()
    assert s.feed("") == []
    assert s.flush() == ""


# -- early first chunk (latency cut for the stream's opening words) -----------


def test_first_chunk_emitted_early_at_clause_boundary() -> None:
    s = SentenceSplitter()
    # Comma after 6 words -> the opening clause is released immediately,
    # punctuation kept, before the sentence has finished streaming.
    chunks = s.feed("The weather in Boston right now, and for")
    assert chunks == ["The weather in Boston right now,"]
    # The rest of that sentence then follows full-sentence behavior.
    assert s.feed(" the rest of the week, it stays sunny. Next") == [
        "and for the rest of the week, it stays sunny."
    ]


def test_first_chunk_clause_needs_six_words() -> None:
    s = SentenceSplitter()
    assert s.feed("Sure, here is what") == []  # comma after only 1 word
    # A later clause mark that does clear the word floor still triggers.
    chunks = s.feed(" I found in the news today, starting with")
    assert chunks == ["Sure, here is what I found in the news today,"]


def test_first_chunk_semicolon_and_emdash_boundaries() -> None:
    s = SentenceSplitter()
    assert s.feed("The report covers three separate topics; each one") == [
        "The report covers three separate topics;"
    ]

    s = SentenceSplitter()
    # Em-dash needs no following whitespace ("great—sunny").
    assert s.feed("The forecast for today looks great—sunny") == [
        "The forecast for today looks great—"
    ]


def test_first_chunk_number_comma_is_not_a_clause() -> None:
    s = SentenceSplitter()
    # "3,500" must not be split even with >= 6 words before the comma.
    assert s.feed("The total comes to about 3,500 dollars") == []


def test_first_chunk_trailing_comma_waits_for_more_input() -> None:
    # Like sentence punctuation, a comma at the very end of the buffer is
    # ambiguous ("3," might continue as "3,500") — wait for the next delta.
    s = SentenceSplitter()
    assert s.feed("The weather in Boston right now,") == []
    assert s.feed(" and more") == ["The weather in Boston right now,"]


def test_first_chunk_forced_after_twelve_words() -> None:
    s = SentenceSplitter()
    text = "one two three four five six seven eight nine ten eleven twelve thirteen"
    # 12 complete words with no boundary -> forced cut at the last whitespace
    # (the trailing token may still be streaming in, so it stays buffered).
    chunks = s.feed(text)
    assert chunks == ["one two three four five six seven eight nine ten eleven twelve"]
    assert s.flush() == "thirteen"


def test_first_chunk_forced_waits_for_complete_words() -> None:
    s = SentenceSplitter()
    # Only 11 complete words (no trailing space -> the 11th token counts,
    # the not-yet-finished 12th would not exist yet).
    assert s.feed("one two three four five six seven eight nine ten eleven ") == []
    assert s.feed("twelve ") == [
        "one two three four five six seven eight nine ten eleven twelve"
    ]
    assert s.flush() == ""


def test_first_full_sentence_in_one_delta_wins_over_early_cut() -> None:
    s = SentenceSplitter()
    # A complete sentence available up front is emitted whole, as before —
    # the early cut only applies while no full sentence is ready.
    chunks = s.feed("It is sunny and warm in Boston today, with a light breeze. More")
    assert chunks == ["It is sunny and warm in Boston today, with a light breeze."]


def test_clause_boundaries_ignored_after_first_emission() -> None:
    s = SentenceSplitter(min_chunk=1)
    assert s.feed("Hello there. Next") == ["Hello there."]
    # >= 6 words before a comma AND >= 12 buffered words, but the first
    # emission already happened: full-sentence behavior only.
    assert s.feed(" up we should look at one two three four five six, and more") == []
    assert s.flush() == "Next up we should look at one two three four five six, and more"


def test_flush_starts_a_new_stream_for_early_first_chunk() -> None:
    s = SentenceSplitter()
    assert s.feed("The weather in Boston right now, and") == [
        "The weather in Boston right now,"
    ]
    s.flush()
    # After flush() the next stream is again eligible for an early cut.
    assert s.feed("The weather in Boston right now, and") == [
        "The weather in Boston right now,"
    ]
