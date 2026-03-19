"""
AgentResponse — the output model produced by each agent in the pipeline.
Each agent receives the previous agent's response as context.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class AgentRole(str, Enum):
    SEARCH   = "search"
    ANSWER   = "answer"
    VERIFY   = "verify"
    RATING   = "rating"


class ResponseStatus(str, Enum):
    SUCCESS   = "success"
    PARTIAL   = "partial"    # Agent completed but with low confidence
    FAILED    = "failed"     # Agent could not produce a usable result
    ESCALATE  = "escalate"   # Human review required


@dataclass
class AgentResponse:
    """Output produced by a single agent step."""

    agent_role: AgentRole
    status: ResponseStatus
    content: str                        # The main output text of this agent
    confidence: float = 1.0            # 0.0 – 1.0
    sources: list[str] = field(default_factory=list)   # URLs or doc refs used
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")

    @property
    def succeeded(self) -> bool:
        return self.status in (ResponseStatus.SUCCESS, ResponseStatus.PARTIAL)

    def __repr__(self):
        preview = self.content[:60] + "..." if len(self.content) > 60 else self.content
        return (
            f"AgentResponse(agent={self.agent_role.value}, "
            f"status={self.status.value}, confidence={self.confidence:.2f}, "
            f"content='{preview}')"
        )
