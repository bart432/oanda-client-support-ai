"""
BaseAgent — abstract base class for all agents in the pipeline.

Every agent must implement the `run` method. Agents communicate by passing
AgentResponse objects — each agent receives the full list of prior responses
so it has complete context of what upstream agents produced.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from src.models.query import ClientQuery
from src.models.response import AgentResponse, AgentRole, ResponseStatus

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for all pipeline agents.

    Subclasses must implement:
        - role (property)  → AgentRole enum value
        - run()            → core agent logic

    Subclasses may override:
        - should_run()     → skip this agent under certain conditions
        - on_error()       → custom error handling / fallback
    """

    def __init__(self, name: Optional[str] = None, max_retries: int = 2):
        self.name = name or self.__class__.__name__
        self.max_retries = max_retries
        self._logger = logging.getLogger(f"agents.{self.name}")

    # ── Abstract interface ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def role(self) -> AgentRole:
        """The AgentRole this agent fulfils in the pipeline."""
        ...

    @abstractmethod
    async def run(
        self,
        query: ClientQuery,
        prior_responses: list[AgentResponse],
    ) -> AgentResponse:
        """
        Execute this agent's core logic.

        Args:
            query:           The original client query.
            prior_responses: Ordered list of responses from upstream agents.

        Returns:
            AgentResponse with this agent's output.
        """
        ...

    # ── Hooks (override as needed) ────────────────────────────────────────────

    def should_run(
        self,
        query: ClientQuery,
        prior_responses: list[AgentResponse],
    ) -> bool:
        """
        Return False to skip this agent entirely.
        Default: always run.
        """
        return True

    async def on_error(
        self,
        query: ClientQuery,
        prior_responses: list[AgentResponse],
        error: Exception,
    ) -> AgentResponse:
        """
        Called when run() raises an exception after all retries.
        Default: return a FAILED response.
        """
        self._logger.error(f"[{self.name}] failed on query {query.query_id}: {error}")
        return AgentResponse(
            agent_role=self.role,
            status=ResponseStatus.FAILED,
            content="",
            confidence=0.0,
            error=str(error),
        )

    # ── Internal execution with retry ────────────────────────────────────────

    async def execute(
        self,
        query: ClientQuery,
        prior_responses: list[AgentResponse],
    ) -> AgentResponse:
        """
        Called by the Orchestrator. Wraps run() with retry logic and error handling.
        Do not override this method.
        """
        if not self.should_run(query, prior_responses):
            self._logger.info(f"[{self.name}] skipped for query {query.query_id}")
            return AgentResponse(
                agent_role=self.role,
                status=ResponseStatus.SUCCESS,
                content="[skipped]",
                confidence=1.0,
                metadata={"skipped": True},
            )

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._logger.info(
                    f"[{self.name}] attempt {attempt}/{self.max_retries} "
                    f"for query {query.query_id}"
                )
                response = await self.run(query, prior_responses)
                self._logger.info(
                    f"[{self.name}] completed — status={response.status.value}, "
                    f"confidence={response.confidence:.2f}"
                )
                return response
            except Exception as exc:
                last_error = exc
                self._logger.warning(
                    f"[{self.name}] attempt {attempt} failed: {exc}"
                )

        return await self.on_error(query, prior_responses, last_error)

    # ── Helpers for subclasses ────────────────────────────────────────────────

    def get_last_response(
        self,
        prior_responses: list[AgentResponse],
        role: AgentRole,
    ) -> Optional[AgentResponse]:
        """Retrieve the most recent response from a specific upstream agent."""
        for r in reversed(prior_responses):
            if r.agent_role == role:
                return r
        return None

    def __repr__(self):
        return f"{self.__class__.__name__}(role={self.role.value})"
