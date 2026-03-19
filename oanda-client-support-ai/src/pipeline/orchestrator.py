"""
Orchestrator — runs the 4-agent pipeline in sequence.

Flow:
    ClientQuery
        │
        ▼
    SearchAgent   → AgentResponse (sources + relevant content)
        │
        ▼
    AnswerAgent   → AgentResponse (synthesised answer)
        │
        ▼
    VerifyAgent   → AgentResponse (verified / corrected answer)
        │
        ▼
    RatingAgent   → RatingResult  (score 1-5 + recommendations)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.models.query import ClientQuery
from src.models.response import AgentResponse, AgentRole, ResponseStatus
from src.models.rating import RatingResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete output of one pipeline run."""

    query: ClientQuery
    responses: list[AgentResponse] = field(default_factory=list)
    rating: Optional[RatingResult] = None
    completed_at: datetime = field(default_factory=datetime.utcnow)
    total_duration_ms: float = 0.0

    @property
    def final_answer(self) -> Optional[str]:
        """The verified answer produced by the VerifyAgent (or AnswerAgent as fallback)."""
        for role in (AgentRole.VERIFY, AgentRole.ANSWER):
            for r in reversed(self.responses):
                if r.agent_role == role and r.succeeded:
                    return r.content
        return None

    @property
    def succeeded(self) -> bool:
        return all(r.succeeded for r in self.responses)

    def __repr__(self):
        score = f"score={self.rating.score}/5" if self.rating else "no-rating"
        return (
            f"PipelineResult(query={self.query.query_id[:8]}, "
            f"{score}, ok={self.succeeded})"
        )


class Orchestrator:
    """
    Runs agents sequentially and collects their responses.

    Usage:
        orchestrator = Orchestrator(
            search_agent=SearchAgent(),
            answer_agent=AnswerAgent(),
            verify_agent=VerifyAgent(),
            rating_agent=RatingAgent(),
        )
        result = await orchestrator.run(ClientQuery("How do I reset my password?"))
    """

    def __init__(
        self,
        search_agent: BaseAgent,
        answer_agent: BaseAgent,
        verify_agent: BaseAgent,
        rating_agent: BaseAgent,
        stop_on_failure: bool = False,
    ):
        self.agents: list[BaseAgent] = [
            search_agent,
            answer_agent,
            verify_agent,
            rating_agent,
        ]
        self.stop_on_failure = stop_on_failure
        self._logger = logging.getLogger("orchestrator")

    async def run(self, query: ClientQuery) -> PipelineResult:
        """
        Execute all agents in sequence for a single ClientQuery.

        Each agent receives all prior AgentResponses as context.
        The RatingAgent's response is parsed into a RatingResult.
        """
        start = datetime.utcnow()
        self._logger.info(f"Pipeline start — query={query.query_id}")

        result = PipelineResult(query=query)

        for agent in self.agents:
            response = await agent.execute(query, result.responses)
            result.responses.append(response)

            if not response.succeeded and self.stop_on_failure:
                self._logger.warning(
                    f"Stopping pipeline early — {agent.name} returned {response.status.value}"
                )
                break

        # Extract RatingResult from the rating agent's response metadata
        rating_response = next(
            (r for r in result.responses if r.agent_role == AgentRole.RATING), None
        )
        if rating_response and "rating_result" in rating_response.metadata:
            result.rating = rating_response.metadata["rating_result"]

        elapsed = (datetime.utcnow() - start).total_seconds() * 1000
        result.total_duration_ms = elapsed
        self._logger.info(
            f"Pipeline complete — query={query.query_id}, "
            f"duration={elapsed:.0f}ms, ok={result.succeeded}"
        )
        return result

    async def run_batch(self, queries: list[ClientQuery]) -> list[PipelineResult]:
        """Run multiple queries concurrently."""
        self._logger.info(f"Batch run — {len(queries)} queries")
        return await asyncio.gather(*[self.run(q) for q in queries])
