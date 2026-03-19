"""
SearchAgent — crawls OANDA Help Center pages to find content relevant
to the client's query.

Crawl strategy:
    1. Any URLs embedded in the query are fetched first as priority sources.
    2. BFS from OANDA_HELP_BASE, following links within help.oanda.com/us/en/
    3. Per-page content is extracted with BeautifulSoup (nav/footer stripped)
    4. The query is expanded with domain synonyms before TF-IDF scoring
    5. Top-k (default 5) results are returned with title, URL, and excerpt

Caching:
    Module-level dict keyed by URL; entries expire after _CACHE_TTL seconds
    so repeated pipeline runs avoid redundant HTTP round-trips.
"""

import asyncio
import math
import re
import time
from collections import defaultdict
from functools import partial
from urllib.parse import urljoin, urlparse

import warnings

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from src.agents.base_agent import BaseAgent
from src.models.query import ClientQuery
from src.models.response import AgentResponse, AgentRole, ResponseStatus

OANDA_HELP_BASE = "https://help.oanda.com/us/en/home.htm"
_ALLOWED_PREFIX = "https://help.oanda.com/us/en/"

_CACHE: dict[str, tuple[float, dict]] = {}  # url -> (monotonic_ts, page_data)
_CACHE_TTL = 3600  # seconds

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; OandaClientSupportBot/1.0; "
        "help.oanda.com crawler)"
    )
}

_MAX_CRAWL_PAGES = 60
_REQUEST_TIMEOUT = 10  # seconds
_EXCERPT_WINDOW = 60   # words
_EXCERPT_MAX_CHARS = 400

_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can",
        "i", "my", "me", "we", "you", "your", "it", "its", "this", "that",
        "s", "re", "ve", "ll", "d", "t",
    }
)

# ── Customer-service synonym expansion ────────────────────────────────────────
# Maps common support intent phrases → additional search tokens to inject.
# Keys are matched case-insensitively as substrings of the query.
_QUERY_EXPANSIONS: list[tuple[str, list[str]]] = [
    # Account access / credentials
    ("password",        ["password", "reset", "login", "credentials", "forgot"]),
    ("reset password",  ["password", "reset", "change", "forgot", "email", "link"]),
    ("forgot password", ["password", "reset", "forgot", "email", "recover"]),
    ("login",           ["login", "sign", "access", "credentials", "account"]),
    ("sign in",         ["login", "sign", "access", "credentials"]),
    ("locked",          ["locked", "blocked", "access", "account", "unlock"]),
    # Account management
    ("open account",    ["open", "account", "register", "signup", "create"]),
    ("close account",   ["close", "account", "terminate", "cancel"]),
    ("verify",          ["verify", "verification", "identity", "kyc", "document"]),
    ("document",        ["document", "upload", "verify", "identity", "proof"]),
    # Deposits / withdrawals
    ("deposit",         ["deposit", "fund", "transfer", "bank", "payment"]),
    ("withdraw",        ["withdraw", "withdrawal", "transfer", "bank", "funds"]),
    ("transfer",        ["transfer", "funds", "deposit", "withdraw", "bank"]),
    # Trading
    ("spread",          ["spread", "cost", "commission", "pricing"]),
    ("leverage",        ["leverage", "margin", "ratio"]),
    ("margin",          ["margin", "leverage", "call", "close"]),
    ("order",           ["order", "trade", "entry", "market", "limit", "stop"]),
    ("platform",        ["platform", "mt4", "mt5", "fxtrade", "app", "download"]),
    # Fees / costs
    ("fee",             ["fee", "cost", "charge", "commission", "pricing"]),
    ("commission",      ["commission", "fee", "cost", "pricing"]),
]

# Regex to extract bare URLs from text
_URL_RE = re.compile(r'https?://[^\s\'"<>]+', re.IGNORECASE)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop words and single chars."""
    return [
        tok
        for tok in re.findall(r"[a-z]+", text.lower())
        if tok not in _STOP_WORDS and len(tok) > 1
    ]


def _extract_excerpt(text: str, query_tokens: list[str]) -> str:
    """
    Slide a window over `text` words and return the window with the highest
    density of query terms, truncated to _EXCERPT_MAX_CHARS.
    """
    words = text.split()
    if not words:
        return ""

    query_set = set(query_tokens)
    best_start, best_score = 0, -1

    for i in range(len(words)):
        chunk = words[i : i + _EXCERPT_WINDOW]
        score = sum(
            1 for w in chunk
            if re.sub(r"[^a-z]", "", w.lower()) in query_set
        )
        if score > best_score:
            best_score, best_start = score, i

    excerpt = " ".join(words[best_start : best_start + _EXCERPT_WINDOW])
    if len(excerpt) > _EXCERPT_MAX_CHARS:
        excerpt = excerpt[:_EXCERPT_MAX_CHARS].rsplit(" ", 1)[0] + "…"
    return excerpt


# ── Agent ─────────────────────────────────────────────────────────────────────

class SearchAgent(BaseAgent):
    """
    Searches OANDA Help Center for content relevant to a client query.

    Returns the top matching excerpts as structured content in AgentResponse,
    with source URLs in the `sources` field.
    """

    def __init__(
        self,
        top_k: int = 5,
        max_pages: int = _MAX_CRAWL_PAGES,
        **kwargs,
    ):
        super().__init__(name="SearchAgent", **kwargs)
        self.top_k = top_k
        self.max_pages = max_pages
        self.base_url = OANDA_HELP_BASE

    @property
    def role(self) -> AgentRole:
        return AgentRole.SEARCH

    async def run(
        self,
        query: ClientQuery,
        prior_responses: list[AgentResponse],
    ) -> AgentResponse:
        """
        Crawl OANDA Help Center (or use cache), rank pages by TF-IDF, and
        return the top-k excerpts with source URLs.

        Priority sources:
            Any URLs found directly in the query text are fetched first and
            pinned at the top of the result list regardless of TF-IDF score.
        """
        self._logger.info(f"Searching for: '{query.text[:80]}'")

        loop = asyncio.get_event_loop()
        results, inline_urls = await loop.run_in_executor(
            None, partial(self._search, query.text)
        )

        if not results:
            return AgentResponse(
                agent_role=self.role,
                status=ResponseStatus.PARTIAL,
                content="No relevant content found on OANDA Help Center.",
                confidence=0.0,
                sources=[self.base_url],
            )

        content = self._format_results(results)
        sources = [r["url"] for r in results]

        return AgentResponse(
            agent_role=self.role,
            status=ResponseStatus.SUCCESS,
            content=content,
            confidence=min(1.0, len(results) / self.top_k),
            sources=sources,
            metadata={
                "num_results": len(results),
                "query": query.text,
                "expanded_query": self._expand_query(query.text),
                "inline_urls": inline_urls,
            },
        )

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def _search(self, query_text: str) -> tuple[list[dict], list[str]]:
        """
        Synchronous: extract inline URLs → crawl → rank → return top-k.

        Returns (results, inline_urls) so the caller can surface which URLs
        were taken directly from the query.
        """
        # 1. Extract any URLs the client embedded in their message
        inline_urls = _URL_RE.findall(query_text)
        priority_pages: list[dict] = []
        priority_urls: set[str] = set()

        for url in inline_urls:
            page = self._fetch_page(url)
            if page and url not in priority_urls:
                priority_pages.append(page)
                priority_urls.add(url)
                self._logger.info(f"Priority source from query URL: {url}")

        # 2. Crawl the help center
        crawled_pages = self._crawl_sitemap()
        if not crawled_pages and not priority_pages:
            self._logger.warning("Crawl returned no pages.")
            return [], inline_urls

        # Merge — priority pages first, then crawled (deduped by URL)
        seen_urls = set(priority_urls)
        all_pages = list(priority_pages)
        for p in crawled_pages:
            if p["url"] not in seen_urls:
                all_pages.append(p)
                seen_urls.add(p["url"])

        # 3. Rank with expanded query
        expanded = self._expand_query(query_text)
        ranked = self._rank_results(expanded, all_pages)

        # 4. Pin priority pages at the top (if they scored > 0), then fill
        priority_scored = [r for r in ranked if r["url"] in priority_urls]
        others = [r for r in ranked if r["url"] not in priority_urls]
        combined = priority_scored + others

        return combined[: self.top_k], inline_urls

    def _expand_query(self, query_text: str) -> str:
        """
        Return an augmented query string with domain-specific synonym tokens
        appended, based on intent phrases found in the original query.
        """
        lower = query_text.lower()
        extra_tokens: list[str] = []
        for phrase, expansions in _QUERY_EXPANSIONS:
            if phrase in lower:
                extra_tokens.extend(expansions)

        if not extra_tokens:
            return query_text

        # Deduplicate while preserving order
        seen: set[str] = set(_tokenize(query_text))
        unique_extras = []
        for tok in extra_tokens:
            if tok not in seen:
                unique_extras.append(tok)
                seen.add(tok)

        return query_text + " " + " ".join(unique_extras)

    def _fetch_page(self, url: str) -> dict | None:
        """
        Fetch `url` and return a page dict::

            {url, title, text, links}

        where `links` is a list of discovered in-scope URLs.
        Results are cached for _CACHE_TTL seconds.
        """
        now = time.monotonic()
        cached = _CACHE.get(url)
        if cached:
            ts, data = cached
            if now - ts < _CACHE_TTL:
                self._logger.debug(f"Cache hit: {url}")
                return data

        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as exc:
            self._logger.warning(f"Fetch failed [{url}]: {exc}")
            return None

        soup = BeautifulSoup(resp.text, "lxml")

        # ── Discover in-scope links before stripping the DOM ─────────────────
        links: list[str] = []
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])
            parsed = urlparse(href)
            # Normalise: drop fragment and query string
            clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if clean.startswith(_ALLOWED_PREFIX) and clean not in links:
                links.append(clean)

        # ── Strip navigation noise ────────────────────────────────────────────
        for tag in soup.select(
            "nav, header, footer, script, style, "
            "[class*='nav'], [class*='menu'], [class*='sidebar'], "
            "[class*='breadcrumb'], [class*='footer'], [class*='header']"
        ):
            tag.decompose()

        # ── Extract title ─────────────────────────────────────────────────────
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url

        # ── Extract main body text ────────────────────────────────────────────
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(
                "div",
                class_=re.compile(r"\b(content|article|body|main)\b", re.I),
            )
            or soup.body
        )
        raw_text = main.get_text(separator=" ", strip=True) if main else ""
        text = re.sub(r"\s+", " ", raw_text).strip()

        data = {"url": url, "title": title, "text": text, "links": links}
        _CACHE[url] = (now, data)
        return data

    def _crawl_sitemap(self) -> list[dict]:
        """
        BFS from `self.base_url`, following links within _ALLOWED_PREFIX.
        Stops when max_pages pages have been fetched.
        """
        visited: set[str] = set()
        queue: list[str] = [self.base_url]
        pages: list[dict] = []

        while queue and len(pages) < self.max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            self._logger.debug(
                f"Crawling ({len(pages) + 1}/{self.max_pages}): {url}"
            )
            page = self._fetch_page(url)
            if not page:
                continue

            pages.append(page)

            for link in page.get("links", []):
                if link not in visited:
                    queue.append(link)

        self._logger.info(f"Crawl complete — {len(pages)} pages fetched.")
        return pages

    def _rank_results(self, query_text: str, pages: list[dict]) -> list[dict]:
        """
        Score each page against `query_text` with TF-IDF (no external deps).

        TF  = normalised term frequency within the page (title × 3 boost)
        IDF = log((N+1) / (df+1)) + 1   (smoothed, always ≥ 1)
        score = Σ TF(t) × IDF(t) for t in query_tokens
        """
        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return []

        N = len(pages)

        # ── Compute TF per page ───────────────────────────────────────────────
        page_tfs: list[dict[str, float]] = []
        df: dict[str, int] = defaultdict(int)

        for page in pages:
            # Title tokens get a 3× weight boost
            tokens = _tokenize(page["title"]) * 3 + _tokenize(page["text"])
            raw_tf: dict[str, float] = defaultdict(float)
            for tok in tokens:
                raw_tf[tok] += 1.0

            max_tf = max(raw_tf.values(), default=1.0)
            tf = {tok: cnt / max_tf for tok, cnt in raw_tf.items()}
            page_tfs.append(tf)

            for tok in set(tokens):
                df[tok] += 1

        # ── Score each page ───────────────────────────────────────────────────
        scored: list[dict] = []
        for page, tf in zip(pages, page_tfs):
            score = sum(
                tf.get(tok, 0.0) * (math.log((N + 1) / (df[tok] + 1)) + 1)
                for tok in query_tokens
            )
            if score <= 0:
                continue

            excerpt = _extract_excerpt(page["text"], query_tokens)
            scored.append(
                {
                    "title": page["title"],
                    "url": page["url"],
                    "excerpt": excerpt or page["text"][:_EXCERPT_MAX_CHARS],
                    "relevance_score": score,
                }
            )

        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored

    # ── Formatting ────────────────────────────────────────────────────────────

    def _format_results(self, results: list[dict]) -> str:
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['title']}")
            lines.append(f"    URL: {r['url']}")
            lines.append(f"    {r['excerpt']}")
            lines.append("")
        return "\n".join(lines).strip()
