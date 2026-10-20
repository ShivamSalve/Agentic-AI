# 📰 Fake News Detector — Multi-Agent System (Agentic AI)

A small multi-agent system that checks whether a news claim is **True / False / Misleading / Unverified**, with evidence and reasoning.

```
Article → Claim Extractor → Evidence Retriever → Verdict Agent → Reviewer Agent → Final Report
                                                      ↑                 │
                                                      └── (if rejected) ┘   ← Reflection loop
```

## 1. Setup (10 minutes)

**Requirements:** Python 3.10+ and an internet connection.

```bash
# 1. Go into the project folder, then install packages
pip install -r requirements.txt

# 2. Create your config file
copy .env.example .env        # Windows
cp .env.example .env          # Mac / Linux

# 3. Get a FREE API key and paste it into .env (as LLM_API_KEY)
#    Groq   : https://console.groq.com/keys
#    Gemini : https://aistudio.google.com/apikey   (see .env.example for its URL/model lines)

# 4. Run the web app
streamlit run app.py
```

> **No key yet?** Set `MOCK_MODE=1` in `.env` to try the interface without API calls. Mock results and the example source are artificial; do not treat them as fact-checks. Evaluation is disabled in mock mode.
> **Model errors?** Free-tier model names change from time to time — check your provider's model list and update `LLM_MODEL` in `.env`.

## 2. Files

| File | What it does |
|---|---|
| `agents.py` | The 4 agents + web-search tool (each agent = one prompt + one function) |
| `workflow.py` | LangGraph workflow connecting the agents, incl. the reviewer loop |
| `app.py` | Streamlit web interface |
| `evaluate.py` | Runs 10 labelled claims, prints accuracy + time, saves `results.csv` |
| `tests/` | Focused checks for agent response validation and the reviewer retry loop |

## 3. Syllabus mapping (use this in your report)

| Concept (syllabus) | Where in the project |
|---|---|
| Multi-agent orchestration | `workflow.py` — LangGraph `StateGraph` with 4 agent nodes |
| ReAct (Reason + Act) | `retrieve_evidence()` — LLM *reasons* a search query, then *acts* by calling the search tool |
| Tool-use pattern | `web_search()` — DuckDuckGo search tool |
| Reflection pattern | Reviewer Agent can reject a verdict → loop back to Verdict Agent (`route_after_review`) |
| Role assignment / delegation | Each agent has its own role prompt and a single responsibility |
| Prompt engineering | Role + strict JSON output prompts in `agents.py` |
| Evaluation (Unit IV) | `evaluate.py` — accuracy and latency; transparency via reasoning + cited sources |

## 4. Getting results for your FA2 report

1. Configure a real model key and set `MOCK_MODE=0`.
2. Run `python evaluate.py` and put the measured accuracy / average-time numbers in your Results section. The script writes `results.csv` to the current directory; do not report mock-mode results as model accuracy.
3. Take screenshots of the app for 2–3 claims (one True, one False, one Misleading).
4. Add 10–20 more claims to `TEST_SET` in `evaluate.py` for a stronger result.
5. Discuss limitations honestly: depends on search quality, LLM can still hallucinate, not suitable for breaking news, cannot check images/videos.

## 5. Easy upgrades (optional)

- Add a **Bias Detector** agent (flags emotional/one-sided language) — copy the pattern of any agent in `agents.py` and add one node in `workflow.py`.
- Swap DuckDuckGo for a **Chroma/FAISS** vector store of trusted fact-check articles (RAG).
- Compare against a traditional classifier (TF-IDF + Logistic Regression on a Kaggle dataset) to prove the "agentic vs traditional" point.

## 6. Troubleshooting

- *"No LLM_API_KEY found"* → you haven't created `.env` or the key line is empty.
- *Everything is "Unverified"* → no relevant search results were available. Check your connection and try again.
- *Rate-limit errors from the LLM* → free tiers have limits; wait or use fewer claims.
- Run the focused tests with `python -m unittest discover -s tests`.

## Important limitations

This is an educational fact-checking demo, not an authoritative news-verification service. Search results are not independently vetted for source quality, the model can misunderstand evidence or invent details, and a verdict is not proof of truth or falsity. Always open and assess the linked sources yourself. Search/API errors are surfaced rather than being treated as an empty evidence set.
