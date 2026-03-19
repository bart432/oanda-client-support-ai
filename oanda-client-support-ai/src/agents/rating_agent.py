"""
RatingAgent — evaluates the entire pipeline run and scores the OANDA
support process on a scale of 1-5, generating actionable recommendations.

Implementation:
    - Sends the full pipeline summary (query → search → answer → verify)
      to Claude with a strict JSON-schema output so the score and lists
      are machine-readable without regex heuristics.
    - The integer score is mapped directly to ProcessScore (IntEnum 1-5).
    - Scores 1-2 are auto-logged as warnings; the Orchestrator can use
      rating.needs_escalation to route them to a human review queue.
    - The RatingResult is stored in AgentResponse.metadata["rating_result"]
      so the Orchestrator can lift it into PipelineResult.rating.
"""

import json

import anthropic
from dotenv import load_dotenv

from src.agents.base_agent import BaseAgent
from src.models.query import ClientQuery
from src.models.rating import ProcessScore, RatingResult
from src.models.response import AgentResponse, AgentRole, ResponseStatus

load_dotenv()

_MAX_TOKENS = 1_024

# ── JSON schema for the structured rating result ──────────────────────────────
_RATING_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5],
            "description": "Overall process score: 1=Critical, 2=Needs Work, 3=Average, 4=Good, 5=Best Practice",
        },
        "justification": {
            "type": "string",
            "description": "Concise explanation of why this score was awarded.",
        },
        "what_works_well": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Aspects of the support process that performed well.",
        },
        "improvements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Significant gaps or issues that need addressing.",
        },
        "quick_wins": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete improvements achievable within one week.",
        },
    },
    "required": ["score", "justification", "what_works_well", "improvements", "quick_wins"],
    "additionalProperties": False,
}

RATING_SYSTEM_PROMPT = """You are a process quality evaluator for OANDA customer support.
Evaluate the entire support interaction — from the initial query through search,
answer, and verification — and rate the overall process on a scale of 1 to 5:

1 = Critical: Process is broken or missing. Immediate action required.
2 = Needs Work: Significant gaps. Confusing or incomplete for clients.
3 = Average: Functional but unimpressive. Room for improvement.
4 = Good: Above average. Minor tweaks would make it excellent.
5 = Best Practice: Exceptional. A benchmark for other processes.

Evaluate across four dimensions:
- Relevance: Did the search surface content that matched the query?
- Accuracy: Is the answer factually correct and grounded in the sources?
- Completeness: Does the answer fully address the client's question?
- Clarity: Is the final answer easy for a client to understand and act on?

Be specific and actionable. Quick wins must be achievable within one week."""


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
        self._client = anthropic.AsyncAnthropic()

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

        pipeline_summary = self._build_summary(query, prior_responses)

        api_response = await self._client.messages.create(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            system=self.system_prompt,
            messages=[{"role": "user", "content": pipeline_summary}],
            output_config={"format": {"type": "json_schema", "schema": _RATING_SCHEMA}},
        )

        raw_json = next(
            (b.text for b in api_response.content if b.type == "text"), "{}"
        )
        result = json.loads(raw_json)

        score = ProcessScore(int(result["score"]))
        rating = RatingResult(
            score=score,
            justification=result.get("justification", ""),
            what_works_well=result.get("what_works_well", []),
            improvements=result.get("improvements", []),
            quick_wins=result.get("quick_wins", []),
            query_id=query.query_id,
        )

        if rating.needs_escalation:
            self._logger.warning(
                f"Low score ({score}/5) for query {query.query_id} — "
                f"flagged for human review. Justification: {rating.justification}"
            )

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
                "input_tokens": api_response.usage.input_tokens,
                "output_tokens": api_response.usage.output_tokens,
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
