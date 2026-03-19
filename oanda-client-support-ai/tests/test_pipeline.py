"""Integration tests for the full pipeline."""

import asyncio
import pytest
from src.models.query import ClientQuery
from src.agents.search_agent import SearchAgent
from src.agents.answer_agent import AnswerAgent
from src.agents.verify_agent import VerifyAgent
from src.agents.rating_agent import RatingAgent
from src.pipeline.orchestrator import Orchestrator


@pytest.fixture
def orchestrator():
    return Orchestrator(
        search_agent=SearchAgent(),
        answer_agent=AnswerAgent(),
        verify_agent=VerifyAgent(),
        rating_agent=RatingAgent(),
    )


@pytest.fixture
def sample_query():
    return ClientQuery(text="How do I reset my OANDA account password?")


class TestOrchestrator:
    def test_run_returns_pipeline_result(self, orchestrator, sample_query):
        result = asyncio.run(orchestrator.run(sample_query))
        assert result is not None
        assert result.query.query_id == sample_query.query_id

    def test_all_four_agents_ran(self, orchestrator, sample_query):
        result = asyncio.run(orchestrator.run(sample_query))
        roles = [r.agent_role.value for r in result.responses]
        assert "search"  in roles
        assert "answer"  in roles
        assert "verify"  in roles
        assert "rating"  in roles

    def test_rating_result_attached(self, orchestrator, sample_query):
        result = asyncio.run(orchestrator.run(sample_query))
        assert result.rating is not None
        assert 1 <= result.rating.score <= 5

    def test_final_answer_not_empty(self, orchestrator, sample_query):
        result = asyncio.run(orchestrator.run(sample_query))
        assert result.final_answer is not None
        assert len(result.final_answer) > 0

    def test_duration_recorded(self, orchestrator, sample_query):
        result = asyncio.run(orchestrator.run(sample_query))
        assert result.total_duration_ms > 0

    def test_batch_run(self, orchestrator):
        queries = [
            ClientQuery(text="How do I deposit funds?"),
            ClientQuery(text="What are OANDA's trading hours?"),
            ClientQuery(text="How do I close a position?"),
        ]
        results = asyncio.run(orchestrator.run_batch(queries))
        assert len(results) == 3
        assert all(r.rating is not None for r in results)
