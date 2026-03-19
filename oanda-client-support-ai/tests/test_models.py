"""Tests for data models."""

import pytest
from src.models.query import ClientQuery
from src.models.response import AgentResponse, AgentRole, ResponseStatus
from src.models.rating import RatingResult, ProcessScore


class TestClientQuery:
    def test_basic_creation(self):
        q = ClientQuery(text="How do I reset my password?")
        assert q.text == "How do I reset my password?"
        assert q.query_id is not None
        assert len(q.query_id) == 36  # UUID format

    def test_strips_whitespace(self):
        q = ClientQuery(text="  hello world  ")
        assert q.text == "hello world"

    def test_empty_text_raises(self):
        with pytest.raises(ValueError):
            ClientQuery(text="")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            ClientQuery(text="   ")

    def test_repr(self):
        q = ClientQuery(text="Short question?")
        assert "Short question?" in repr(q)


class TestAgentResponse:
    def test_basic_creation(self):
        r = AgentResponse(
            agent_role=AgentRole.SEARCH,
            status=ResponseStatus.SUCCESS,
            content="Found relevant content.",
        )
        assert r.agent_role == AgentRole.SEARCH
        assert r.succeeded is True

    def test_failed_response_not_succeeded(self):
        r = AgentResponse(
            agent_role=AgentRole.ANSWER,
            status=ResponseStatus.FAILED,
            content="",
            confidence=0.0,
        )
        assert r.succeeded is False

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            AgentResponse(
                agent_role=AgentRole.VERIFY,
                status=ResponseStatus.SUCCESS,
                content="ok",
                confidence=1.5,
            )

    def test_partial_is_succeeded(self):
        r = AgentResponse(
            agent_role=AgentRole.SEARCH,
            status=ResponseStatus.PARTIAL,
            content="Partial results.",
        )
        assert r.succeeded is True


class TestRatingResult:
    def test_score_labels(self):
        r = RatingResult(score=ProcessScore.BEST_PRACTICE, justification="Excellent.")
        assert r.label == "Best Practice"
        assert r.color == "#00E676"

    def test_escalation_flag(self):
        r1 = RatingResult(score=ProcessScore.CRITICAL, justification="Broken.")
        r2 = RatingResult(score=ProcessScore.GOOD, justification="Good.")
        assert r1.needs_escalation is True
        assert r2.needs_escalation is False

    def test_summary_contains_score(self):
        r = RatingResult(
            score=ProcessScore.AVERAGE,
            justification="Room for improvement.",
            improvements=["Fix response time"],
            quick_wins=["Add FAQ section"],
        )
        summary = r.summary()
        assert "3/5" in summary
        assert "Average" in summary
        assert "Fix response time" in summary
