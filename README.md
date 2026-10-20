# 📰 Fake News Detector — Multi-Agent System (Agentic AI)

A small multi-agent system that checks whether a news claim is **True / False / Misleading / Unverified**, with evidence and reasoning.

```
Article → Claim Extractor → Evidence Retriever → Verdict Agent → Reviewer Agent → Final Report
                                                      ↑                 │
                                                      └── (if rejected) ┘   ← Reflection loop
```

## 1. Setup (10 minutes)

**Requirements:** Python 3.10+, Ollama, and an internet connection for web evidence searches.

```bash
# 1. Go into the project folder, then install packages
pip install -r requirements.txt

# 2. Install Ollama from https://ollama.com/download
#    Then download the default model:
ollama pull llama3.2

# 3. Create your config file
copy .env.example .env        # Windows
cp .env.example .env          # Mac / Linux

# 4. Ensure Ollama is running, then launch the app
streamlit run app.py
```

Ollama runs locally at `http://localhost:11434` by default and needs no API key. Choose **Ask Ollama** to generate a response to your own question or custom prompt. Choose **Fact-check a claim/article** to run the multi-agent evidence workflow on your own text; web evidence search requires internet access.

> Set `MOCK_MODE=1` in `.env` to try the interface with artificial responses and sources. Mock output is not a real answer or fact-check.
> To use another Ollama model, pull it with `ollama pull <model-name>` and set `LLM_MODEL` to that model name in `.env`.
> If Ollama reports GPU/CUDA out-of-memory errors, try a smaller model. On Windows, you can run a CPU-only server from a separate PowerShell window with `$env:OLLAMA_HOST='127.0.0.1:11435'; $env:OLLAMA_LLM_LIBRARY='cpu'; ollama serve`, then set `LLM_BASE_URL=http://127.0.0.1:11435/v1` in `.env`.

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

- *Ollama memory/CUDA error* → use a smaller model or the CPU-only server command above.
- *"Connection error" with Ollama* → make sure Ollama is running and the model is downloaded (`ollama list`).
- *Everything is "Unverified"* → no relevant search results were available. Check your connection and try again.
- *Rate-limit errors from the LLM* → free tiers have limits; wait or use fewer claims.
- Run the focused tests with `python -m unittest discover -s tests`.

## Important limitations

This is an educational fact-checking demo, not an authoritative news-verification service. Search results are not independently vetted for source quality, the model can misunderstand evidence or invent details, and a verdict is not proof of truth or falsity. Always open and assess the linked sources yourself. Search/API errors are surfaced rather than being treated as an empty evidence set.
