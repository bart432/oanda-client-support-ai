# OANDA-Client Support AI

A network of four specialized AI agents that automatically handles OANDA client support queries, scores each process 1–5, and generates improvement recommendations.

## Architecture

```
ClientQuery
    │
    ▼
SearchAgent   → Scans OANDA Help Center for relevant content
    │
    ▼
AnswerAgent   → Synthesises a clear, client-ready answer (Claude API)
    │
    ▼
VerifyAgent   → Checks accuracy, completeness & compliance
    │
    ▼
RatingAgent   → Scores the process 1-5 + improvement recommendations
    │
    ▼
PipelineResult (final answer + rating)
```

## Project Structure

```
src/
├── models/          # Data models: ClientQuery, AgentResponse, RatingResult
├── agents/          # BaseAgent + 4 agent implementations
│   ├── base_agent.py
│   ├── search_agent.py
│   ├── answer_agent.py
│   ├── verify_agent.py
│   └── rating_agent.py
├── pipeline/
│   └── orchestrator.py   # Runs agents in sequence, handles retries
└── utils/
    └── logger.py

tests/
├── test_models.py
└── test_pipeline.py
```

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/bart432/oanda-client-support-ai.git
cd oanda-client-support-ai

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 5. Run tests
pytest tests/ -v
```

## Usage

```python
import asyncio
from src.models.query import ClientQuery
from src.agents.search_agent import SearchAgent
from src.agents.answer_agent import AnswerAgent
from src.agents.verify_agent import VerifyAgent
from src.agents.rating_agent import RatingAgent
from src.pipeline.orchestrator import Orchestrator

async def main():
    orchestrator = Orchestrator(
        search_agent=SearchAgent(),
        answer_agent=AnswerAgent(),
        verify_agent=VerifyAgent(),
        rating_agent=RatingAgent(),
    )

    query = ClientQuery(text="How do I reset my OANDA account password?")
    result = await orchestrator.run(query)

    print(result.final_answer)
    print(result.rating.summary())

asyncio.run(main())
```

## Rating Scale

| Score | Label | Meaning |
|-------|-------|---------|
| 1 | Critical | Broken or missing — immediate action required |
| 2 | Needs Work | Significant gaps — confusing or incomplete |
| 3 | Average | Functional but room for improvement |
| 4 | Good | Above average — minor tweaks needed |
| 5 | Best Practice | Exceptional — a benchmark for others |

## Development Phases

- **Phase 1 (Architecture)** ✅ — Models, BaseAgent, Orchestrator, stubs
- **Phase 2 (Build)** 🔄 — Implement real SearchAgent crawler + Claude API calls
- **Phase 3 (Test & Ship)** ⏳ — 50-query pilot, A/B test, GitHub PR

## Built With

- [Anthropic Claude](https://anthropic.com) — LLM backbone
- [Claude Code](https://claude.ai/claude-code) — AI-driven development
- [Asana](https://asana.com) — Task tracking
- [GitHub](https://github.com) — Version control & PRs
