"""
RatingResult — the final output of the Rating Agent.
Contains a 1-5 score, detailed justification, and actionable recommendations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Optional


class ProcessScore(IntEnum):
    CRITICAL       = 1   # Broken or missing — immediate action required
    NEEDS_WORK     = 2   # Significant gaps — confusing or incomplete
    AVERAGE        = 3   # Functional but unimpressive — room for improvement
    GOOD           = 4   # Above average — minor tweaks needed
    BEST_PRACTICE  = 5   # Exceptional — a benchmark for other processes


SCORE_LABELS = {
    ProcessScore.CRITICAL:      "Critical",
    ProcessScore.NEEDS_WORK:    "Needs Work",
    ProcessScore.AVERAGE:       "Average",
    ProcessScore.GOOD:          "Good",
    ProcessScore.BEST_PRACTICE: "Best Practice",
}

SCORE_COLORS = {
    ProcessScore.CRITICAL:      "#FF3D3D",
    ProcessScore.NEEDS_WORK:    "#FF8C00",
    ProcessScore.AVERAGE:       "#FFD700",
    ProcessScore.GOOD:          "#7ED321",
    ProcessScore.BEST_PRACTICE: "#00E676",
}


@dataclass
class RatingResult:
    """Final rating produced by the Rating Agent for a completed pipeline run."""

    score: ProcessScore
    justification: str                          # Why this score was given
    what_works_well: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    quick_wins: list[str] = field(default_factory=list)   # Fixes achievable in < 1 week
    query_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def label(self) -> str:
        return SCORE_LABELS[self.score]

    @property
    def color(self) -> str:
        return SCORE_COLORS[self.score]

    @property
    def needs_escalation(self) -> bool:
        """Flag queries rated 1-2 for human review."""
        return self.score <= ProcessScore.NEEDS_WORK

    def summary(self) -> str:
        lines = [
            f"Score: {self.score}/5 — {self.label}",
            f"Justification: {self.justification}",
        ]
        if self.what_works_well:
            lines.append("What works well: " + "; ".join(self.what_works_well))
        if self.improvements:
            lines.append("Improvements needed: " + "; ".join(self.improvements))
        if self.quick_wins:
            lines.append("Quick wins: " + "; ".join(self.quick_wins))
        return "\n".join(lines)

    def __repr__(self):
        return f"RatingResult(score={self.score}/5 [{self.label}], query_id={self.query_id})"
