"""
VerifyAgent — checks the AnswerAgent's response for accuracy, completeness,
and compliance before it reaches the client.

TODO (Claude Code — Phase 2):
    - Implement LLM-based fact-checking against SearchAgent sources
    - Add OANDA-specific compliance rules (e.g. no financial advice)
    - Implement confidence scoring based on source coverage
    - Add contradiction detection between answer and source material
"""

from src.agents.base_agent import BaseAgent
from src.models.query import ClientQuery
from src.models.response import AgentResponse, AgentRole, ResponseStatus

VERIFY_SYSTEM_PROMPT = """You are a strict quality assurance agent for OANDA support.
Your job is to verify that the answer provided is:
1. Factually accurate based on the source material
2. Complete — it addresses all parts of the client's question
3. Compliant — it does not provide personalised financial advice
4. Clear — it is written in plain language a client can understand

If the answer passes all checks, return it with any minor improvements.
If it fails, correct it or flag it for human review."""


class VerifyAgent(BaseAgent):
    """
    Verifies and optionally corrects the AnswerAgent's response.
    Flags answers requiring human review as ESCALATE.
    """

    def __init__(self, model: str = "claude-opus-4-6", **kwargs):
        super().__init__(name="VerifyAgent", **kwargs)
        self.model = model
        self.system_prompt = VERIFY_SYSTEM_PROMPT

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
        Verify the AnswerAgent's output and return a corrected/approved response.
        """
        answer_response = self.get_last_response(prior_responses, AgentRole.ANSWER)
        search_response = self.get_last_response(prior_responses, AgentRole.SEARCH)

        answer_text = answer_response.content if answer_response else ""
        source_text = search_response.content if search_response else ""

        self._logger.info(f"Verifying answer for query: '{query.text[:80]}'")

        # ── STUB — replace with real verification logic in Phase 2 ────────────
        verified_text, confidence, status = self._stub_verify(answer_text, source_text)
        # ─────────────────────────────────────────────────────────────────────

        return AgentResponse(
            agent_role=self.role,
            status=status,
            content=verified_text,
            confidence=confidence,
            sources=answer_response.sources if answer_response else [],
            metadata={
                "model": self.model,
                "original_answer_length": len(answer_text),
                "verified": True,
            },
        )

    # ── Private helpers (stubs) ───────────────────────────────────────────────

    def _stub_verify(
        self, answer: str, sources: str
    ) -> tuple[str, float, ResponseStatus]:
        """
        Placeholder verification. Replace with real LLM-based check.
        Returns: (verified_text, confidence, status)
        """
        verified = (
            f"[VERIFIED — STUB]\n{answer}\n\n"
            f"Note: Real verification against OANDA sources will be implemented in Phase 2."
        )
        return verified, 0.9, ResponseStatus.SUCCESS
