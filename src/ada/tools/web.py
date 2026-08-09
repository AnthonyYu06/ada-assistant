"""Web tools: DuckDuckGo search and readable page fetching."""

from __future__ import annotations

import html
import ipaddress
import logging
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

from .registry import ToolError, ToolResult, tool

log = logging.getLogger("ada.tools.web")

_MAX_BODY_BYTES = 2_000_000
_MAX_TEXT_CHARS = 4000
_MAX_CARD_CHARS = 1500
_USER_AGENT = "Mozilla/5.0 (Ada assistant)"
_MAX_REDIRECTS = 3
_BLOCKED_HOST_MESSAGE = "I won't fetch that address."


def _host_is_blocked(hostname: str) -> bool:
    """True if the host resolves to any loopback/private/link-local/reserved/
    multicast/unspecified address (SSRF guard). IP-literal hosts are checked
    directly. Unresolvable hosts fail closed (blocked)."""
    if not hostname:
        return True
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return True
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr.split("%", 1)[0])
        except ValueError:
            return True
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return True
    return False


class _TextExtractor(HTMLParser):
    """Extract readable text from HTML, skipping non-content elements."""

    _SKIP = {"script", "style", "noscript", "nav", "header", "footer", "template", "svg"}
    _BLOCK = {
        "p", "div", "br", "li", "ul", "ol", "tr", "td", "th", "table",
        "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "aside",
        "main", "blockquote", "pre", "figure", "figcaption", "hr", "form",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            if self._skip_depth > 0:
                self._skip_depth -= 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        raw = re.sub(r" ?\n ?", "\n", raw)
        raw = re.sub(r"\n{2,}", "\n", raw)
        return raw.strip()


def _domain(url: str) -> str:
    netloc = urlparse(url).netloc
    return netloc.removeprefix("www.")


def _page_title(raw_html: str) -> str:
    """The page's <title> text, collapsed to one line ('' if absent)."""
    match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return " ".join(html.unescape(match.group(1)).split())


@tool(
    name="search_web",
    description=(
        "Search the web for current information you don't know or that may "
        "have changed. Use for news, prices, weather, sports scores, recent "
        "events, and anything after your training cutoff. Returns titles, "
        "snippets, and domains; use fetch_url to read a result in full."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "max_results": {
                "type": "integer",
                "description": "How many results to return (1-8).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
    risk="safe",
    category="web",
)
def search_web(query: str, max_results: int = 5) -> str | ToolResult:
    try:
        from ddgs import DDGS  # type: ignore[import-not-found]
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore[import-not-found]
        except ImportError:
            raise ToolError("Web search isn't installed — run: pip install ddgs") from None

    if not query or not query.strip():
        raise ToolError("Tell me what to search for.")
    try:
        limit = max(1, min(8, int(max_results)))
    except (TypeError, ValueError):
        limit = 5

    try:
        results = list(DDGS().text(query.strip(), max_results=limit))
    except Exception as exc:  # noqa: BLE001 - network/service errors
        log.warning("Web search failed: %s", exc)
        raise ToolError("Web search failed — the search service may be unavailable.") from exc

    if not results:
        return "No web results found."

    lines: list[str] = []
    card_results: list[dict] = []
    for i, result in enumerate(results[:limit], start=1):
        title = " ".join((result.get("title") or "Untitled").split())
        body = " ".join((result.get("body") or "").split())
        if len(body) > 150:
            body = body[:147].rstrip() + "..."
        href = result.get("href") or result.get("url") or ""
        domain = _domain(href) if href else ""
        line = f"{i}. {title}: {body}" if body else f"{i}. {title}"
        if domain:
            line += f" ({domain})"
        lines.append(line)
        entry: dict = {"title": title, "url": href}
        if body:
            entry["snippet"] = body
        card_results.append(entry)
    return ToolResult(
        speech="\n".join(lines),
        display={
            "kind": "web_results",
            "title": query.strip(),
            "results": card_results,
        },
    )


@tool(
    name="fetch_url",
    description=(
        "Fetch a web page or JSON API over http/https and return its readable "
        "text (up to a few thousand characters). Use after search_web to read "
        "a promising result, or when the user gives you a specific URL."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Full URL to fetch, e.g. 'https://example.com/article'.",
            }
        },
        "required": ["url"],
    },
    risk="safe",
    category="web",
)
def fetch_url(url: str) -> str | ToolResult:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ToolError("I can only fetch http or https URLs.")
    if not parsed.netloc:
        raise ToolError("That doesn't look like a valid web address.")

    current = url
    chunks: list[bytes] = []
    content_type = ""
    encoding = "utf-8"
    try:
        # Redirects are handled manually so the SSRF host check can run on
        # every hop, not just the first URL.
        for _hop in range(_MAX_REDIRECTS + 1):
            hop_parsed = urlparse(current)
            if hop_parsed.scheme not in ("http", "https"):
                raise ToolError(_BLOCKED_HOST_MESSAGE)
            if _host_is_blocked(hop_parsed.hostname or ""):
                raise ToolError(_BLOCKED_HOST_MESSAGE)
            with httpx.stream(
                "GET",
                current,
                follow_redirects=False,
                timeout=10,
                headers={"User-Agent": _USER_AGENT},
            ) as response:
                if 300 <= response.status_code < 400:
                    location = response.headers.get("location")
                    if not location:
                        raise ToolError("That page redirected without a destination.")
                    current = urljoin(current, location)
                    continue
                if response.status_code >= 400:
                    raise ToolError(f"That page returned an error, status {response.status_code}.")
                content_type = (response.headers.get("content-type") or "").lower()
                if not (content_type.startswith("text/") or "json" in content_type):
                    raise ToolError("That page isn't text — I can only read text or JSON pages.")
                total = 0
                for chunk in response.iter_bytes():
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= _MAX_BODY_BYTES:
                        break
                encoding = response.encoding or "utf-8"
                break
        else:
            raise ToolError("That page redirected too many times.")
    except ToolError:
        raise
    except httpx.TimeoutException:
        raise ToolError("That page took too long to load.") from None
    except httpx.HTTPError as exc:
        log.debug("fetch_url(%s) failed: %s", url, exc)
        raise ToolError("I couldn't reach that page.") from exc

    body = b"".join(chunks)[:_MAX_BODY_BYTES]
    try:
        text = body.decode(encoding, errors="replace")
    except LookupError:
        text = body.decode("utf-8", errors="replace")

    page_title = ""
    if "html" in content_type:
        page_title = _page_title(text)
        extractor = _TextExtractor()
        extractor.feed(text)
        text = extractor.text()
    else:
        text = text.strip()

    if not text:
        return "That page didn't contain any readable text."
    if len(text) > _MAX_TEXT_CHARS:
        text = text[:_MAX_TEXT_CHARS].rstrip() + "... (truncated)"
    excerpt = text
    if len(excerpt) > _MAX_CARD_CHARS:
        excerpt = excerpt[:_MAX_CARD_CHARS].rstrip() + "…"
    return ToolResult(
        speech=text,
        display={
            "kind": "markdown",
            "title": page_title or _domain(current) or _domain(url),
            "body": excerpt,
        },
    )
