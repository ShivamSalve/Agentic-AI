"""app.py  --  Simple web UI.  Run with:  streamlit run app.py"""
import os

import streamlit as st

from agents import AgentError, MOCK
from workflow import NODE_LABELS, build_graph

st.set_page_config(page_title="Fake News Detector - Agentic AI", page_icon="📰", layout="wide")
st.title("📰 Fake News Detector")
st.caption("A multi-agent system: Claim Extractor → Evidence Retriever → Verdict Agent → Reviewer Agent")

if MOCK:
    st.info("MOCK MODE is ON - no real AI is used. Set MOCK_MODE=0 in your .env file for real results.")
elif not os.getenv("LLM_API_KEY", "").strip() or os.getenv("LLM_API_KEY", "").strip().lower() in {
    "paste_your_key_here", "your_api_key",
}:
    st.error("No LLM_API_KEY found. Copy `.env.example` to `.env` and add your API key.")
    st.stop()

EXAMPLES = {
    "Great Wall myth": "The Great Wall of China is the only man-made structure visible from the Moon with the naked eye.",
    "Health rumour": "Drinking hot water with lemon cures cancer, and doctors are hiding this from the public.",
    "True claim": "Mount Everest is the highest mountain above sea level, and it lies on the border of Nepal and China.",
}

st.write("**Try an example or paste your own:**")
cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(cols, EXAMPLES.items()):
    if col.button(label, use_container_width=True):
        st.session_state["article"] = text

article = st.text_area("News headline / article / claim", key="article", height=150)

ICONS = {"True": "✅", "False": "❌", "Misleading": "⚠️", "Unverified": "❓"}

if st.button("🔍 Verify", type="primary", disabled=not article.strip()):
    graph = build_graph()
    final = {"article": article, "retries": 0, "log": []}

    try:
        with st.status("Agents are working...", expanded=True) as status:
            for step in graph.stream(final, stream_mode="updates"):
                for node, update in step.items():
                    final.update(update)
                    st.write(f"**{NODE_LABELS[node]}** → {update['log'][-1]}")
            status.update(label="Done!", state="complete")
    except AgentError as e:
        st.error(str(e))
        st.stop()

    overall = final["overall"]
    banner = f"{ICONS[overall]} Overall verdict: **{overall}**"
    {"True": st.success, "False": st.error, "Misleading": st.warning, "Unverified": st.info}[overall](banner)
    st.caption("AI-generated assessments can be wrong. Review the cited sources before relying on a verdict.")

    st.subheader("Claim-by-claim breakdown")
    for r in final["results"]:
        ev = final["evidence"][r["claim"]]
        with st.expander(f"{ICONS[r['verdict']]} {r['claim']}", expanded=True):
            c1, c2 = st.columns([1, 3])
            c1.metric("Verdict", r["verdict"])
            c1.progress(r["confidence"] / 100, text=f"Confidence: {r['confidence']}%")
            c2.write(f"**Reasoning:** {r['reasoning']}")
            c2.caption(f"Reviewer: {'approved ✔' if r['approved'] else 'not approved'} - {r['feedback']}")
            c2.caption(f"Search query used: _{ev['query']}_")
            for i, e in enumerate(ev["results"], 1):
                title = e.get("title") or "Untitled source"
                url = e.get("url")
                c2.markdown(f"[{i}] [{title}]({url})" if url else f"[{i}] {title}")
                if e.get("snippet"):
                    c2.caption(e["snippet"])
