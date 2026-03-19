"""
OANDA Support AI — Streamlit frontend.

Usage:
    streamlit run app.py          # local
    docker compose up             # Docker
"""

import asyncio

import streamlit as st

from src.agents.answer_agent import AnswerAgent
from src.agents.rating_agent import RatingAgent
from src.agents.search_agent import SearchAgent
from src.agents.verify_agent import VerifyAgent
from src.models.query import ClientQuery
from src.models.response import AgentRole, ResponseStatus
from src.pipeline.orchestrator import Orchestrator

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="OANDA Support AI",
    page_icon="🏦",
    layout="centered",
)

st.title("🏦 OANDA Support AI")
st.caption("Searches the OANDA Help Center and delivers a verified, rated answer.")

# ── Orchestrator (cached — shared across reruns) ──────────────────────────────

@st.cache_resource
def get_orchestrator() -> Orchestrator:
    return Orchestrator(
        search_agent=SearchAgent(),
        answer_agent=AnswerAgent(),
        verify_agent=VerifyAgent(),
        rating_agent=RatingAgent(),
    )

orchestrator = get_orchestrator()

# ── Query form ────────────────────────────────────────────────────────────────

with st.form("query_form"):
    query_text = st.text_area(
        "Your question",
        placeholder="How do I reset my OANDA account password?",
        height=100,
    )
    submitted = st.form_submit_button("Ask", type="primary")

# ── Run pipeline ──────────────────────────────────────────────────────────────

if submitted and query_text.strip():
    query = ClientQuery(text=query_text.strip())

    with st.spinner("Searching OANDA Help Center and generating answer…"):
        result = asyncio.run(orchestrator.run(query))

    # Helper: pull a response by role
    def _resp(role: AgentRole):
        return next((r for r in result.responses if r.agent_role == role), None)

    search_resp = _resp(AgentRole.SEARCH)
    verify_resp = _resp(AgentRole.VERIFY)
    rating_resp = _resp(AgentRole.RATING)

    # ── Search sources ────────────────────────────────────────────────────────
    if search_resp and search_resp.sources:
        with st.expander(f"🔍 {len(search_resp.sources)} sources found", expanded=False):
            for url in search_resp.sources:
                st.markdown(f"- [{url}]({url})")

    # ── Final answer ──────────────────────────────────────────────────────────
    final_answer = result.final_answer

    if final_answer:
        # Compliance flag
        if verify_resp and verify_resp.status == ResponseStatus.ESCALATE:
            st.warning(
                "⚠️ This answer has been flagged for human review "
                "and may require additional verification."
            )

        st.markdown("### Answer")
        st.markdown(final_answer)

        # Verify badge
        if verify_resp:
            verdict = verify_resp.metadata.get("verdict", "")
            conf = verify_resp.confidence
            emoji = {"pass": "✅", "corrected": "🔧", "escalate": "⚠️"}.get(verdict, "")
            issues = verify_resp.metadata.get("issues", [])
            caption = f"{emoji} **{verdict.capitalize()}** · confidence {conf:.0%}"
            if issues:
                caption += f" · {len(issues)} issue(s) corrected"
            st.caption(caption)
    else:
        st.error("Could not generate an answer. Please try rephrasing your question.")

    # ── Rating ────────────────────────────────────────────────────────────────
    if result.rating:
        r = result.rating
        st.markdown("---")
        col_score, col_detail = st.columns([1, 4])

        with col_score:
            st.markdown(
                f"<div style='text-align:center'>"
                f"<span style='font-size:3rem;color:{r.color}'><b>{r.score}/5</b></span><br>"
                f"<span style='color:{r.color}'>{r.label}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with col_detail:
            st.markdown(r.justification)

            if r.quick_wins:
                with st.expander("💡 Quick wins"):
                    for w in r.quick_wins:
                        st.markdown(f"- {w}")

            if r.improvements:
                with st.expander("🔧 Improvements"):
                    for imp in r.improvements:
                        st.markdown(f"- {imp}")
