"""
AnswerAgent — uses Gemini via Vertex AI to synthesise a clear, client-ready
answer from the content retrieved by the SearchAgent.

Implementation notes:
    - Uses google-genai SDK with Vertex AI backend (ADC, no API key needed).
    - GCP project and region are read from GCP_PROJECT / GCP_REGION env vars
      (loaded from .env via python-dotenv).
    - Context is capped at _MAX_CONTEXT_CHARS so the prompt stays well within
      Gemini's context window.
    - If the query itself contains substantial inline text or URLs, that content
      is surfaced as a primary context section before the search results.
    - If search confidence is low, the agent signals this in the answer so the
      VerifyAgent / RatingAgent can act accordingly.
    - If query.language is not "en", the system prompt instructs Gemini to
      reply in that language.
"""

import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import FinishReason

from src.agents.base_agent import BaseAgent
from src.models.query import ClientQuery
from src.models.response import AgentResponse, AgentRole, ResponseStatus

load_dotenv()

_GCP_PROJECT = os.getenv("GCP_PROJECT", "")
_GCP_REGION = os.getenv("GCP_REGION", "us-central1")
_MAX_CONTEXT_CHARS = 8_000
_MAX_TOKENS = 1_024

# Queries longer than this are assumed to contain pasted / inline text
_INLINE_TEXT_THRESHOLD = 300  # characters
# Minimum search confidence below which we warn the model
_LOW_SEARCH_CONFIDENCE = 0.4

_URL_RE = re.compile(r'https?://[^\s\'"<>]+', re.IGNORECASE)

SYSTEM_PROMPT = """You are a helpful OANDA support specialist.
Your job is to answer client questions clearly and accurately.

When answering:
1. If "Client-provided information" is present, treat it as the primary,
   authoritative source — the client may have pasted content or shared a URL.
2. Use the "OANDA Help Center excerpts" as supporting evidence.
3. If search confidence is flagged as LOW, acknowledge any uncertainty and
   advise the client to verify with OANDA support if needed.
4. Be concise, friendly, and precise.
5. If neither source contains enough information, say so honestly and suggest
   the client contacts OANDA support directly."""


class AnswerAgent(BaseAgent):
    """
    Synthesises the SearchAgent's results into a clear client-facing answer
    using Gemini via Vertex AI.

    Enhancement: checks for embedded URLs / inline text in the query itself
    and uses them as a primary context section before falling back to search.
    """

    def __init__(self, model: str = "gemini-2.5-flash", **kwargs):
        super().__init__(name="AnswerAgent", **kwargs)
        self.model = model
        self.system_prompt = SYSTEM_PROMPT
        self._client = genai.Client(
            vertexai=True, project=_GCP_PROJECT, location=_GCP_REGION
        )

    @property
    def role(self) -> AgentRole:
        return AgentRole.ANSWER

    def should_run(
        self,
        query: ClientQuery,
        prior_responses: list[AgentResponse],
    ) -> bool:
        """Run if SearchAgent succeeded OR the query itself contains inline content."""
        search = self.get_last_response(prior_responses, AgentRole.SEARCH)
        has_search = search is not None and search.succeeded
        has_inline = self._extract_inline_context(query.text) is not None
        return has_search or has_inline

    async def run(
        self,
        query: ClientQuery,
        prior_responses: list[AgentResponse],
    ) -> AgentResponse:
        """Generate an answer using Gemini, grounded in search + inline context."""
        search_response = self.get_last_response(prior_responses, AgentRole.SEARCH)
        search_context = search_response.content if search_response else ""
        search_confidence = search_response.confidence if search_response else 0.0
        sources = search_response.sources if search_response else []

        self._logger.info(f"Generating answer for query: '{query.text[:80]}'")

        # ── Check for inline context in the query itself ──────────────────────
        inline_context = self._extract_inline_context(query.text)
        low_search = search_confidence < _LOW_SEARCH_CONFIDENCE

        # ── Truncate search context to fit token budget ───────────────────────
        truncated = len(search_context) > _MAX_CONTEXT_CHARS
        if truncated:
            search_context = search_context[:_MAX_CONTEXT_CHARS].rsplit("\n", 1)[0]

        system = self._build_system_prompt(query.language)
        user_message = self._build_user_message(
            query.text, search_context, inline_context, truncated, low_search
        )

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=_MAX_TOKENS,
            ),
        )

        answer = response.text or ""
        finish = response.candidates[0].finish_reason if response.candidates else None
        # Reduce confidence when search was weak and no inline context existed
        base_confidence = 0.9 if finish == FinishReason.STOP else 0.6
        confidence = base_confidence if (not low_search or inline_context) else base_confidence * 0.7

        return AgentResponse(
            agent_role=self.role,
            status=ResponseStatus.SUCCESS,
            content=answer,
            confidence=confidence,
            sources=sources,
            metadata={
                "model": self.model,
                "context_length": len(search_context),
                "context_truncated": truncated,
                "low_search_confidence": low_search,
                "has_inline_context": inline_context is not None,
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
                "finish_reason": str(finish),
            },
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_inline_context(self, query_text: str) -> str | None:
        """
        Return inline context if the query appears to contain embedded
        information (URLs or substantial pasted text), else None.
        """
        # Case 1: explicit URLs in the query
        urls = _URL_RE.findall(query_text)
        if urls:
            return f"URLs referenced by client: {', '.join(urls)}"

        # Case 2: long query — likely pasted text / email excerpt
        stripped = query_text.strip()
        if len(stripped) >= _INLINE_TEXT_THRESHOLD:
            return f"Client-provided text:\n{stripped}"

        return None

    def _build_system_prompt(self, language: str) -> str:
        if language and language.lower() != "en":
            return (
                self.system_prompt
                + f"\n\nIMPORTANT: The client's preferred language is '{language}'. "
                  "Respond in that language."
            )
        return self.system_prompt

    def _build_user_message(
        self,
        query_text: str,
        search_context: str,
        inline_context: str | None,
        truncated: bool,
        low_search: bool,
    ) -> str:
        parts: list[str] = []

        if inline_context:
            parts.append("## Client-provided information\n" + inline_context)

        if search_context:
            note = " [truncated]" if truncated else ""
            confidence_note = " ⚠️ Search confidence is LOW." if low_search else ""
            parts.append(
                f"## OANDA Help Center excerpts{note}{confidence_note}\n"
                + search_context
            )
        elif not inline_context:
            parts.append("## OANDA Help Center excerpts\n(no results found)")

        parts.append(f"## Client question\n{query_text}")

        return "\n\n---\n\n".join(parts)
