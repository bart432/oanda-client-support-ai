"""
AnswerAgent — uses the Claude API to synthesise a clear, client-ready
answer from the content retrieved by the SearchAgent.

TODO (Claude Code — Phase 2):
    - Connect to Anthropic SDK (anthropic.AsyncAnthropic)
    - Design and iterate on the system prompt
    - Handle token limits for long search results
    - Add multi-language support (query.language)
"""

from src.agents.base_agent import BaseAgent
from src.models.query import ClientQuery
from src.models.response import AgentResponse, AgentRole, ResponseStatus

SYSTEM_PROMPT = """You are a helpful OANDA support specialist.
Your job is to answer client questions clearly and accurately, based ONLY
on the OANDA Help Center content provided. Be concise, friendly, and precise.
If the provided content does not contain enough information to answer the question,
say so honestly and suggest the client contacts OANDA support directly."""


class AnswerAgent(BaseAgent):
    """
    Synthesises the SearchAgent's results into a clear client-facing answer
    using the Claude API.
    """

    def __init__(self, model: str = "claude-opus-4-6", **kwargs):
        super().__init__(name="AnswerAgent", **kwargs)
        self.model = model
        self.system_prompt = SYSTEM_PROMPT

    @property
    def role(self) -> AgentRole:
        return AgentRole.ANSWER

    def should_run(
        self,
        query: ClientQuery,
        prior_responses: list[AgentResponse],
    ) -> bool:
        """Skip if SearchAgent found nothing useful."""
        search = self.get_last_response(prior_responses, AgentRole.SEARCH)
        return search is not None and search.succeeded

    async def run(
        self,
        query: ClientQuery,
        prior_responses: list[AgentResponse],
    ) -> AgentResponse:
        """
        Generate an answer using Claude, grounded in SearchAgent's output.
        """
        search_response = self.get_last_response(prior_responses, AgentRole.SEARCH)
        context = search_response.content if search_response else "(no search results)"

        self._logger.info(f"Generating answer for query: '{query.text[:80]}'")

        # ── STUB — replace with real Anthropic SDK call in Phase 2 ───────────
        answer = self._stub_answer(query.text, context)
        sources = search_response.sources if search_response else []
        # ─────────────────────────────────────────────────────────────────────

        return AgentResponse(
            agent_role=self.role,
            status=ResponseStatus.SUCCESS,
            content=answer,
            confidence=0.85,
            sources=sources,
            metadata={"model": self.model, "context_length": len(context)},
        )

    # ── Private helpers (stubs) ───────────────────────────────────────────────

    def _stub_answer(self, query_text: str, context: str) -> str:
        """
        Placeholder answer. Replace with real Claude API call:

            client = anthropic.AsyncAnthropic()
            message = await client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.system_prompt,
                messages=[{"role": "user", "content": f"Question: {query_text}\n\nContext:\n{context}"}],
            )
            return message.content[0].text
        """
        return (
            f"[STUB ANSWER] Based on the OANDA Help Center, here is the answer "
            f"to '{query_text}':\n\n"
            f"This placeholder will be replaced by a real Claude API response in Phase 2.\n\n"
            f"Context received from SearchAgent:\n{context[:200]}..."
        )
