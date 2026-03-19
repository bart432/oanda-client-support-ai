"""
VerifyAgent — checks the AnswerAgent's response for accuracy, completeness,
and compliance before it reaches the client.

Uses Gemini via Vertex AI with structured JSON output (response_mime_type +
response_schema) so the verdict and confidence are machine-readable without
string parsing.
"""

import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.agents.base_agent import BaseAgent
from src.models.query import ClientQuery
from src.models.response import AgentResponse, AgentRole, ResponseStatus

load_dotenv()

_GCP_PROJECT = os.getenv("GCP_PROJECT", "")
_GCP_REGION = os.getenv("GCP_REGION", "us-central1")
_MAX_SOURCE_CHARS = 6_000
_MAX_TOKENS = 2_048

_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["pass", "corrected", "escalate"],
            "description": (
                "'pass' = answer is accurate and compliant as-is; "
                "'corrected' = answer needed fixes (return improved version); "
                "'escalate' = answer cannot be safely delivered to the client"
            ),
        },
        "verified_answer": {
            "type": "string",
            "description": (
                "The final answer text. If verdict is 'pass', copy the original. "
                "If 'corrected', provide the improved version. "
                "If 'escalate', explain what is wrong and suggest the client "
                "contacts OANDA support directly."
            ),
        },
        "confidence": {
            "type": "number",
            "description": "Confidence 0.0–1.0 that the verified answer is accurate and complete.",
        },
        "issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of issues found (empty when verdict is 'pass').",
        },
    },
    "required": ["verdict", "verified_answer", "confidence", "issues"],
}

VERIFY_SYSTEM_PROMPT = """You are a strict quality assurance agent for OANDA support.
Your job is to verify that the draft answer is:
1. Factually accurate — every claim is supported by the provided source material
2. Complete — it addresses all parts of the client's question
3. Compliant — it does NOT provide personalised financial advice, make guarantees
   about returns, or recommend specific trades or instruments
4. Clear — it is written in plain language a client can understand

Rules:
- If the answer passes all four checks, set verdict to "pass" and copy it unchanged.
- If it has minor factual errors or omissions you can fix, set verdict to "corrected"
  and provide the improved answer.
- If it contains financial advice, guarantees, severe inaccuracies, or cannot be
  corrected safely, set verdict to "escalate" and explain what went wrong."""


class VerifyAgent(BaseAgent):
    """
    Verifies and optionally corrects the AnswerAgent's response.
    Flags answers requiring human review with ResponseStatus.ESCALATE.
    """

    def __init__(self, model: str = "gemini-2.5-flash", **kwargs):
        super().__init__(name="VerifyAgent", **kwargs)
        self.model = model
        self.system_prompt = VERIFY_SYSTEM_PROMPT
        self._client = genai.Client(
            vertexai=True, project=_GCP_PROJECT, location=_GCP_REGION
        )

    @property
    def role(self) -> AgentRole:
        return AgentRole.VERIFY

    def should_run(
        self,
        query: ClientQuery,
        prior_responses: list[AgentResponse],
    ) -> bool:
        """Only run if we have a valid answer to verify."""
        answer = self.get_last_response(prior_responses, AgentRole.ANSWER)
        return answer is not None and answer.succeeded

    async def run(
        self,
        query: ClientQuery,
        prior_responses: list[AgentResponse],
    ) -> AgentResponse:
        """
        Verify the AnswerAgent's output against the original source material.
        Uses structured JSON output so verdict/confidence are parsed directly.
        """
        answer_response = self.get_last_response(prior_responses, AgentRole.ANSWER)
        search_response = self.get_last_response(prior_responses, AgentRole.SEARCH)

        answer_text = answer_response.content if answer_response else ""
        source_text = search_response.content if search_response else "(no source material)"

        self._logger.info(f"Verifying answer for query: '{query.text[:80]}'")

        if len(source_text) > _MAX_SOURCE_CHARS:
            source_text = source_text[:_MAX_SOURCE_CHARS].rsplit("\n", 1)[0]

        user_message = self._build_user_message(query.text, source_text, answer_text)

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                max_output_tokens=_MAX_TOKENS,
                response_mime_type="application/json",
                response_schema=_VERIFY_SCHEMA,
            ),
        )

        if not response.text:
            raise ValueError("Empty response from model — will retry")
        result = json.loads(response.text)

        verdict = result.get("verdict", "escalate")
        verified_answer = result.get("verified_answer", answer_text)
        confidence = float(max(0.0, min(1.0, result.get("confidence", 0.5))))
        issues = result.get("issues", [])

        status = (
            ResponseStatus.ESCALATE if verdict == "escalate" else ResponseStatus.SUCCESS
        )

        if verdict == "escalate":
            self._logger.warning(
                f"Answer escalated for query {query.query_id}: {issues}"
            )
        elif issues:
            self._logger.info(
                f"Answer corrected ({len(issues)} issue(s)) for query {query.query_id}"
            )

        return AgentResponse(
            agent_role=self.role,
            status=status,
            content=verified_answer,
            confidence=confidence,
            sources=answer_response.sources if answer_response else [],
            metadata={
                "model": self.model,
                "verdict": verdict,
                "issues": issues,
                "original_answer_length": len(answer_text),
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
            },
        )

    def _build_user_message(
        self, query_text: str, source_text: str, answer_text: str
    ) -> str:
        return (
            f"ORIGINAL CLIENT QUESTION:\n{query_text}\n\n"
            f"SOURCE MATERIAL FROM OANDA HELP CENTER:\n---\n{source_text}\n---\n\n"
            f"DRAFT ANSWER TO VERIFY:\n---\n{answer_text}\n---"
        )
