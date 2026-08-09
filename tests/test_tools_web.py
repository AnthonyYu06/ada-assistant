"""Tests for ada.tools.web: HTML text extraction, fetch_url, search_web.

All network access is mocked; the optional `ddgs` dependency is simulated
via sys.modules injection so these tests run whether or not it is installed.
"""

from __future__ import annotations

import socket
import sys
import types
from typing import Any

import httpx
import pytest

from ada.tools.registry import ToolError, ToolResult
from ada.tools.web import _TextExtractor, fetch_url, search_web


@pytest.fixture(autouse=True)
def _resolve_to_public_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    """By default, make every host resolve to a public IP so the SSRF guard
    lets fetch_url proceed. Individual tests override this to test refusal."""

    def fake_getaddrinfo(host: str, *args: Any, **kwargs: Any):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

_HTML = """
<html>
  <head><title>Ignored Title Tag</title><style>.c { color: red }</style></head>
  <body>
    <nav>Navigation SKIPME</nav>
    <header>Header SKIPME</header>
    <h1>Big Announcement</h1>
    <script>var secret = "SKIPME";</script>
    <p>First paragraph with a &amp; entity.</p>
    <div>Second block<br>with a break.</div>
    <footer>Footer SKIPME</footer>
  </body>
</html>
"""


# -- _TextExtractor -----------------------------------------------------------

def test_text_extractor_keeps_content_and_skips_chrome() -> None:
    extractor = _TextExtractor()
    extractor.feed(_HTML)
    text = extractor.text()
    assert "Big Announcement" in text
    assert "First paragraph with a & entity." in text  # charrefs decoded
    assert "Second block" in text
    assert "with a break." in text
    assert "SKIPME" not in text  # script/style/nav/header/footer dropped
    assert ".c { color: red }" not in text
    assert "\n\n" not in text  # blank runs collapsed


def test_text_extractor_block_tags_separate_lines() -> None:
    extractor = _TextExtractor()
    extractor.feed("<p>one</p><p>two</p>")
    assert extractor.text() == "one\ntwo"


# -- fetch_url ----------------------------------------------------------------

class _FakeResponse:
    def __init__(
        self,
        body: bytes = b"",
        content_type: str = "text/html; charset=utf-8",
        status_code: int = 200,
        encoding: str = "utf-8",
    ) -> None:
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self.encoding = encoding
        self._body = body

    def iter_bytes(self):
        yield self._body


def _install_fake_stream(
    monkeypatch: pytest.MonkeyPatch, response: _FakeResponse
) -> dict[str, Any]:
    """Replace httpx.stream with a fake; returns the captured call info."""
    captured: dict[str, Any] = {}

    class _Ctx:
        def __enter__(self) -> _FakeResponse:
            return response

        def __exit__(self, *exc: object) -> bool:
            return False

    def fake_stream(method: str, url: str, **kwargs: Any) -> _Ctx:
        captured["method"] = method
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _Ctx()

    monkeypatch.setattr(httpx, "stream", fake_stream)
    return captured


def test_fetch_url_rejects_non_http_schemes() -> None:
    for url in ("ftp://example.com/file", "file:///etc/passwd", "javascript:x"):
        with pytest.raises(ToolError, match="http"):
            fetch_url(url)


def test_fetch_url_rejects_missing_host() -> None:
    with pytest.raises(ToolError, match="valid web address"):
        fetch_url("https://")


def test_fetch_url_extracts_html_text(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_stream(
        monkeypatch, _FakeResponse(body=_HTML.encode("utf-8"))
    )
    result = fetch_url("https://example.com/article")
    assert isinstance(result, ToolResult)
    text = result.speech
    assert "Big Announcement" in text
    assert "First paragraph with a & entity." in text
    assert "SKIPME" not in text
    # The markdown card carries the page title and a text excerpt.
    assert result.display is not None
    assert result.display["kind"] == "markdown"
    assert result.display["title"] == "Ignored Title Tag"
    assert "Big Announcement" in result.display["body"]
    # Sane request settings.
    assert captured["method"] == "GET"
    assert captured["url"] == "https://example.com/article"
    # Redirects are followed manually so the SSRF host check runs on each hop.
    assert captured["kwargs"]["follow_redirects"] is False
    assert "Ada" in captured["kwargs"]["headers"]["User-Agent"]


def test_fetch_url_returns_json_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_stream(
        monkeypatch,
        _FakeResponse(
            body=b'{"temperature": 21.5}\n',
            content_type="application/json",
        ),
    )
    result = fetch_url("https://api.example.com/weather")
    assert isinstance(result, ToolResult)
    assert result.speech == '{"temperature": 21.5}'
    # No <title> in JSON — the card falls back to the domain.
    assert result.display["kind"] == "markdown"
    assert result.display["title"] == "api.example.com"


def test_fetch_url_http_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_stream(monkeypatch, _FakeResponse(status_code=404))
    with pytest.raises(ToolError, match="404"):
        fetch_url("https://example.com/missing")


def test_fetch_url_rejects_binary_content(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_stream(
        monkeypatch,
        _FakeResponse(body=b"\x89PNG...", content_type="image/png"),
    )
    with pytest.raises(ToolError, match="text"):
        fetch_url("https://example.com/logo.png")


def test_fetch_url_truncates_long_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    body = ("<p>" + "word " * 2000 + "</p>").encode("utf-8")
    _install_fake_stream(monkeypatch, _FakeResponse(body=body))
    result = fetch_url("https://example.com/long")
    assert isinstance(result, ToolResult)
    text = result.speech
    assert text.endswith("... (truncated)")
    assert len(text) <= 4000 + len("... (truncated)")
    # The card excerpt is capped harder (~1500 chars).
    assert len(result.display["body"]) <= 1500 + 1


def test_fetch_url_empty_page(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_stream(
        monkeypatch,
        _FakeResponse(body=b"<script>var x = 1;</script>"),
    )
    assert fetch_url("https://example.com/empty") == (
        "That page didn't contain any readable text."
    )


def test_fetch_url_refuses_loopback_host(monkeypatch: pytest.MonkeyPatch) -> None:
    def resolve_loopback(host: str, *args: Any, **kwargs: Any):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve_loopback)

    # httpx must never be hit for a blocked host.
    def explode(*args: Any, **kwargs: Any):
        raise AssertionError("httpx.stream should not be called for a blocked host")

    monkeypatch.setattr(httpx, "stream", explode)
    for url in ("http://localhost/admin", "http://127.0.0.1:8080/"):
        with pytest.raises(ToolError, match="won't fetch"):
            fetch_url(url)


def test_fetch_url_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_timeout(*args: Any, **kwargs: Any):
        raise httpx.TimeoutException("too slow")

    monkeypatch.setattr(httpx, "stream", raise_timeout)
    with pytest.raises(ToolError, match="too long"):
        fetch_url("https://example.com/slow")


def test_fetch_url_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_connect(*args: Any, **kwargs: Any):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(httpx, "stream", raise_connect)
    with pytest.raises(ToolError, match="reach"):
        fetch_url("https://example.com/down")


# -- search_web ---------------------------------------------------------------

def _install_fake_ddgs(
    monkeypatch: pytest.MonkeyPatch, results: list[dict[str, str]]
) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    module = types.ModuleType("ddgs")

    class DDGS:
        def text(self, query: str, max_results: int = 5):
            captured["query"] = query
            captured["max_results"] = max_results
            return list(results)

    module.DDGS = DDGS  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ddgs", module)
    return captured


def test_search_web_raises_install_hint_when_ddgs_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # None in sys.modules makes `import ddgs` raise ImportError even when
    # the package is actually installed.
    monkeypatch.setitem(sys.modules, "ddgs", None)
    monkeypatch.setitem(sys.modules, "duckduckgo_search", None)
    with pytest.raises(ToolError, match="pip install ddgs"):
        search_web("anything")


def test_search_web_formats_results(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_ddgs(
        monkeypatch,
        [
            {
                "title": "Python  (programming language)",
                "body": "A high-level programming language.",
                "href": "https://www.python.org/about",
            },
            {"title": "Py docs", "body": "", "href": ""},
        ],
    )
    out = search_web("python language", max_results=2)
    assert isinstance(out, ToolResult)
    lines = out.speech.splitlines()
    assert lines[0] == (
        "1. Python (programming language): "
        "A high-level programming language. (python.org)"
    )
    assert lines[1] == "2. Py docs"
    assert captured["query"] == "python language"
    assert captured["max_results"] == 2
    # The web_results card mirrors the results.
    card = out.display
    assert card["kind"] == "web_results"
    assert card["title"] == "python language"
    assert card["results"][0] == {
        "title": "Python (programming language)",
        "url": "https://www.python.org/about",
        "snippet": "A high-level programming language.",
    }
    assert card["results"][1] == {"title": "Py docs", "url": ""}


def test_search_web_truncates_long_snippets(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_ddgs(
        monkeypatch,
        [{"title": "T", "body": "b" * 400, "href": "https://example.com"}],
    )
    out = search_web("query")
    assert isinstance(out, ToolResult)
    assert "..." in out.speech
    assert "b" * 200 not in out.speech
    assert "b" * 200 not in out.display["results"][0]["snippet"]


def test_search_web_clamps_max_results(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_ddgs(monkeypatch, [])
    search_web("query", max_results=99)
    assert captured["max_results"] == 8
    search_web("query", max_results=-3)
    assert captured["max_results"] == 1


def test_search_web_empty_query(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_ddgs(monkeypatch, [])
    with pytest.raises(ToolError, match="search for"):
        search_web("   ")


def test_search_web_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_ddgs(monkeypatch, [])
    assert search_web("obscure query") == "No web results found."


def test_search_web_service_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("ddgs")

    class DDGS:
        def text(self, query: str, max_results: int = 5):
            raise RuntimeError("rate limited")

    module.DDGS = DDGS  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ddgs", module)
    with pytest.raises(ToolError, match="Web search failed"):
        search_web("query")
