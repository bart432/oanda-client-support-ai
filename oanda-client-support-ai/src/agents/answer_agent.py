"""
AnswerAgent — uses the Claude API to synthesise a clear, client-ready
answer from the content retrieved by the SearchAgent.

Implementation notes:
    - Uses anthropic.AsyncAnthropic with streaming to avoid HTTP timeouts
      on long answers and to support models with large max_tokens.
    - Context is capped at _MAX_CONTEXT_CHARS (~2 000 tokens) so that the
      search excerpts, system prompt, and full answer all fit comfortably
      inside the model's context window.
    - If query.language is not "en", an instruction to reply in that
      language is appended to the system prompt.
    - ANTHROPIC_API_KEY is read from the environment; python-dotenv loads
      .env automatically so local runs work without exporting the variable.
"""

import anthropic
from dotenv import load_dotenv

from src.agents.base_agent import BaseAgent
from src.models.query import ClientQuery
from src.models.response import AgentResponse, AgentRole, ResponseStatus

load_dotenv()

_MAX_CONTEXT_CHARS = 8_000   # ~2 000 tokens — leaves headroom for prompt + reply
_MAX_TOKENS = 1_024           # maximum answer length

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
        self._client = anthropic.AsyncAnthropic()   # reads ANTHROPIC_API_KEY from env

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

        Uses streaming + get_final_message() to stay within HTTP timeouts
        even when max_tokens is large.
        """
        search_response = self.get_last_response(prior_responses, AgentRole.SEARCH)
        context = search_response.content if search_response else "(no search results)"
        sources = search_response.sources if search_response else []

        self._logger.info(f"Generating answer for query: '{query.text[:80]}'")

        # ── Truncate context if needed ────────────────────────────────────────
        truncated = len(context) > _MAX_CONTEXT_CHARS
        if truncated:
            context = context[:_MAX_CONTEXT_CHARS].rsplit("\n", 1)[0]
            self._logger.debug(
                f"Context truncated to {len(context)} chars for query {query.query_id}"
            )

        system = self._build_system_prompt(query.language)
        user_message = self._build_user_message(query.text, context, truncated)

        # ── Stream the response (avoids timeout on long answers) ──────────────
        async with self._client.messages.stream(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            final = await stream.get_final_message()

        answer = next(
            (block.text for block in final.content if block.type == "text"), ""
        )
        confidence = 0.9 if final.stop_reason == "end_turn" else 0.6

        return AgentResponse(
            agent_role=self.role,
            status=ResponseStatus.SUCCESS,
            content=answer,
            confidence=confidence,
            sources=sources,
            metadata={
                "model": self.model,
                "context_length": len(context),
                "context_truncated": truncated,
                "input_tokens": final.usage.input_tokens,
                "output_tokens": final.usage.output_tokens,
                "stop_reason": final.stop_reason,
            },
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_system_prompt(self, language: str) -> str:
        if language and language.lower() != "en":
            return (
                self.system_prompt
                + f"\n\nIMPORTANT: The client's preferred language is '{language}'. "
                  "Respond in that language."
            )
        return self.system_prompt

    def _build_user_message(
        self, query_text: str, context: str, truncated: bool
    ) -> str:
        note = " [truncated to fit context window]" if truncated else ""
        return (
            f"The following excerpts were retrieved from the OANDA Help Center{note}:\n\n"
            f"---\n{context}\n---\n\n"
            f"Client question: {query_text}"
        )
