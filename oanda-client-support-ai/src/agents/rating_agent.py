"""
RatingAgent — evaluates the entire pipeline run and scores the OANDA
support process on a scale of 1-5, generating actionable recommendations.

TODO (Claude Code — Phase 2):
    - Implement LLM-based holistic evaluation of the full pipeline run
    - Design rubric prompts for each score level (1-5)
    - Parse LLM output into a structured RatingResult
    - Track score trends over time (store in DB or JSON log)
    - Auto-escalate score 1-2 results to human review queue
"""

from src.agents.base_agent import BaseAgent
from src.models.query import ClientQuery
from src.models.response import AgentResponse, AgentRole, ResponseStatus
from src.models.rating import RatingResult, ProcessScore

RATING_SYSTEM_PROMPT = """You are a process quality evaluator for OANDA customer support.
Evaluate the entire support interaction — from the initial query through search,
answer, and verification — and rate the overall process on a scale of 1 to 5:

1 = Critical: Process is broken or missing. Immediate action required.
2 = Needs Work: Significant gaps. Confusing or incomplete for clients.
3 = Average: Functional but unimpressive. Room for improvement.
4 = Good: Above average. Minor tweaks would make it excellent.
5 = Best Practice: Exceptional. A benchmark for other processes.

Return a structured evaluation with:
- score (1-5)
- justification (why this score)
- what_works_well (list)
- improvements (list)
- quick_wins (list of fixes achievable within 1 week)"""


class RatingAgent(BaseAgent):
    """
    Evaluates the full pipeline run and produces a RatingResult (score 1-5).

    The RatingResult is stored in AgentResponse.metadata["rating_result"]
    so the Orchestrator can extract it into PipelineResult.rating.
    """

    def __init__(self, model: str = "claude-opus-4-6", **kwargs):
        super().__init__(name="RatingAgent", **kwargs)
        self.model = model
        self.system_prompt = RATING_SYSTEM_PROMPT

    @property
    def role(self) -> AgentRole:
        return AgentRole.RATING

    async def run(
        self,
        query: ClientQuery,
        prior_responses: list[AgentResponse],
    ) -> AgentResponse:
        """
        Evaluate the full pipeline run and return a score + recommendations.
        """
        self._logger.info(f"Rating pipeline run for query: '{query.text[:80]}'")

        # Build a summary of everything that happened
        pipeline_summary = self._build_summary(query, prior_responses)

        # ── STUB — replace with real LLM evaluation in Phase 2 ───────────────
        rating = self._stub_rating(query, prior_responses)
        # ─────────────────────────────────────────────────────────────────────

        rating.query_id = query.query_id

        return AgentResponse(
            agent_role=self.role,
            status=ResponseStatus.SUCCESS,
            content=rating.summary(),
            confidence=1.0,
            metadata={
                "rating_result": rating,
                "model": self.model,
                "pipeline_summary_length": len(pipeline_summary),
                "needs_escalation": rating.needs_escalation,
            },
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_summary(
        self, query: ClientQuery, prior_responses: list[AgentResponse]
    ) -> str:
        """Build a text summary of the full pipeline run for the LLM to evaluate."""
        parts = [f"Client Query: {query.text}\n"]
        for r in prior_responses:
            parts.append(f"--- {r.agent_role.value.upper()} AGENT ---")
            parts.append(f"Status: {r.status.value} | Confidence: {r.confidence:.2f}")
            parts.append(r.content[:500])
            parts.append("")
        return "\n".join(parts)

    def _stub_rating(
        self, query: ClientQuery, prior_responses: list[AgentResponse]
    ) -> RatingResult:
        """
        Placeholder rating. Replace with real LLM evaluation in Phase 2.

        Real implementation will:
            1. Call Claude with pipeline_summary + RATING_SYSTEM_PROMPT
            2. Parse structured JSON output into RatingResult fields
        """
        all_succeeded = all(r.succeeded for r in prior_responses)
        avg_confidence = (
            sum(r.confidence for r in prior_responses) / len(prior_responses)
            if prior_responses else 0.0
        )

        if all_succeeded and avg_confidence >= 0.8:
            score = ProcessScore.GOOD
        elif all_succeeded:
            score = ProcessScore.AVERAGE
        else:
            score = ProcessScore.NEEDS_WORK

        return RatingResult(
            score=score,
            justification=(
                f"[STUB] Pipeline completed with avg confidence {avg_confidence:.2f}. "
                f"Real LLM-based evaluation will be implemented in Phase 2."
            ),
            what_works_well=["Pipeline executed end-to-end without crashing"],
            improvements=["Replace all stubs with real implementations"],
            quick_wins=["Add Anthropic API key to .env and implement AnswerAgent first"],
        )
