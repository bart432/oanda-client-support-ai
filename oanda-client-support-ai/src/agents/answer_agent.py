"""
AnswerAgent — uses Gemini via Vertex AI to synthesise a clear, client-ready
answer from the content retrieved by the SearchAgent.

Implementation notes:
    - Uses google-genai SDK with Vertex AI backend (ADC, no API key needed).
    - GCP project and region are read from GCP_PROJECT / GCP_REGION env vars
      (loaded from .env via python-dotenv).
    - Context is capped at _MAX_CONTEXT_CHARS so the prompt stays well within
      Gemini's context window.
    - If query.language is not "en", the system prompt instructs Gemini to
      reply in that language.
"""

import os

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

SYSTEM_PROMPT = """You are a helpful OANDA support specialist.
Your job is to answer client questions clearly and accurately, based ONLY
on the OANDA Help Center content provided. Be concise, friendly, and precise.
If the provided content does not contain enough information to answer the question,
say so honestly and suggest the client contacts OANDA support directly."""


class AnswerAgent(BaseAgent):
    """
    Synthesises the SearchAgent's results into a clear client-facing answer
    using Gemini via Vertex AI.
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
        """Skip if SearchAgent found nothing useful."""
        search = self.get_last_response(prior_responses, AgentRole.SEARCH)
        return search is not None and search.succeeded

    async def run(
        self,
        query: ClientQuery,
        prior_responses: list[AgentResponse],
    ) -> AgentResponse:
        """Generate an answer using Gemini, grounded in SearchAgent's output."""
        search_response = self.get_last_response(prior_responses, AgentRole.SEARCH)
        context = search_response.content if search_response else "(no search results)"
        sources = search_response.sources if search_response else []

        self._logger.info(f"Generating answer for query: '{query.text[:80]}'")

        truncated = len(context) > _MAX_CONTEXT_CHARS
        if truncated:
            context = context[:_MAX_CONTEXT_CHARS].rsplit("\n", 1)[0]

        system = self._build_system_prompt(query.language)
        user_message = self._build_user_message(query.text, context, truncated)

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
        confidence = 0.9 if finish == FinishReason.STOP else 0.6

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
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
                "finish_reason": str(finish),
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
