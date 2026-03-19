"""
SearchAgent — crawls OANDA Help Center pages to find content relevant
to the client's query.

TODO (Claude Code — Phase 2):
    - Implement _fetch_page() using requests + BeautifulSoup
    - Implement _rank_results() using TF-IDF or embedding similarity
    - Implement _crawl_sitemap() to discover all help.oanda.com pages
    - Add caching layer (TTL-based) to avoid redundant fetches
    - Handle pagination and nested help articles
"""

from src.agents.base_agent import BaseAgent
from src.models.query import ClientQuery
from src.models.response import AgentResponse, AgentRole, ResponseStatus

OANDA_HELP_BASE = "https://help.oanda.com/us/en/home.htm"


class SearchAgent(BaseAgent):
    """
    Searches OANDA Help Center for content relevant to a client query.

    Returns the top matching excerpts as structured content in AgentResponse,
    with source URLs in the `sources` field.
    """

    def __init__(self, top_k: int = 5, **kwargs):
        super().__init__(name="SearchAgent", **kwargs)
        self.top_k = top_k          # Number of top results to return
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
        Search OANDA Help Center for content relevant to query.text.

        Current implementation: stub — returns placeholder content.
        Claude Code will implement the full crawler in Phase 2.
        """
        self._logger.info(f"Searching for: '{query.text[:80]}'")

        # ── STUB — replace with real crawler in Phase 2 ──────────────────────
        results = self._stub_search(query.text)
        # ─────────────────────────────────────────────────────────────────────

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
            metadata={"num_results": len(results), "query": query.text},
        )

    # ── Private helpers (stubs) ───────────────────────────────────────────────

    def _stub_search(self, query_text: str) -> list[dict]:
        """
        Placeholder search. Replace with real implementation.
        Returns list of dicts: {title, url, excerpt, relevance_score}
        """
        return [
            {
                "title": f"OANDA Help: {query_text[:40]}",
                "url": self.base_url,
                "excerpt": (
                    f"[STUB] This is where the relevant OANDA help content "
                    f"for '{query_text}' will appear after the crawler is implemented."
                ),
                "relevance_score": 0.9,
            }
        ]

    def _format_results(self, results: list[dict]) -> str:
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['title']}")
            lines.append(f"    URL: {r['url']}")
            lines.append(f"    {r['excerpt']}")
            lines.append("")
        return "\n".join(lines).strip()
