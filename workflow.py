"""
workflow.py  --  Connects the agents into a LangGraph workflow.

    extract -> retrieve -> verdict -> review --(all approved / retry limit)--> report -> END
                              ^                |
                              +---(rejected)---+          <- Reflection loop
"""
from typing import TypedDict

from langgraph.graph import END, StateGraph

from agents import AgentError, extract_claims, judge_claim, retrieve_evidence, review_verdict

MAX_RETRIES = 1  # how many times the Reviewer may send verdicts back for revision

NODE_LABELS = {
    "extract": "Claim Extractor",
    "retrieve": "Evidence Retriever",
    "verdict": "Verdict Agent",
    "review": "Reviewer Agent",
    "report": "Final Report",
}


class State(TypedDict, total=False):
    article: str
    claims: list      # ["claim 1", ...]
    evidence: dict    # claim -> {"query": str, "results": [..]}
    results: list     # [{claim, verdict, confidence, reasoning, approved, feedback}]
    retries: int
    overall: str
    log: list         # human-readable trace shown in the UI


# ----------------------------------------------------------------------------
# Nodes (each node = one agent step)
# ----------------------------------------------------------------------------
def extract_node(state: State):
    claims = extract_claims(state["article"])
    return {"claims": claims, "log": state["log"] + [f"Found {len(claims)} claim(s) to check."]}


def retrieve_node(state: State):
    evidence = {}
    for claim in state["claims"]:
        query, results = retrieve_evidence(claim)
        evidence[claim] = {"query": query, "results": results}
    n = sum(len(v["results"]) for v in evidence.values())
    return {"evidence": evidence, "log": state["log"] + [f"Searched the web and collected {n} source(s)."]}


def verdict_node(state: State):
    previous = {r["claim"]: r for r in state.get("results", [])}
    results = []
    for claim in state["claims"]:
        old = previous.get(claim)
        feedback = old["feedback"] if old and not old["approved"] else None  # use reviewer feedback if any
        verdict = judge_claim(claim, state["evidence"][claim]["results"], feedback)
        results.append({"claim": claim, **verdict, "approved": False, "feedback": ""})
    tag = " (revised after reviewer feedback)" if state.get("retries", 0) > 0 else ""
    return {"results": results, "log": state["log"] + [f"Drafted verdicts{tag}."]}


def review_node(state: State):
    results = []
    for r in state["results"]:
        review = review_verdict(r["claim"], state["evidence"][r["claim"]]["results"], r)
        results.append({**r, **review})
    rejected = sum(1 for r in results if not r["approved"])
    msg = "Approved all verdicts." if rejected == 0 else f"Rejected {rejected} verdict(s) - sending back for revision."
    return {"results": results, "retries": state.get("retries", 0) + 1, "log": state["log"] + [msg]}


def report_node(state: State):
    verdicts = [r["verdict"] for r in state["results"]]
    if "False" in verdicts:
        overall = "False"
    elif "Misleading" in verdicts:
        overall = "Misleading"
    elif all(v == "True" for v in verdicts):
        overall = "True"
    else:
        overall = "Unverified"
    return {"overall": overall, "log": state["log"] + [f"Overall verdict: {overall}."]}


def route_after_review(state: State) -> str:
    """Reflection loop: go back to the Verdict Agent if something was rejected (max MAX_RETRIES times)."""
    all_ok = all(r["approved"] for r in state["results"])
    if all_ok or state["retries"] > MAX_RETRIES:
        return "report"
    return "verdict"


# ----------------------------------------------------------------------------
# Build the graph
# ----------------------------------------------------------------------------
def build_graph():
    g = StateGraph(State)
    g.add_node("extract", extract_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("verdict", verdict_node)
    g.add_node("review", review_node)
    g.add_node("report", report_node)

    g.set_entry_point("extract")
    g.add_edge("extract", "retrieve")
    g.add_edge("retrieve", "verdict")
    g.add_edge("verdict", "review")
    g.add_conditional_edges("review", route_after_review, {"verdict": "verdict", "report": "report"})
    g.add_edge("report", END)
    return g.compile()


def run(article: str) -> dict:
    """Run the whole pipeline and return the final state."""
    if not isinstance(article, str) or not article.strip():
        raise AgentError("Enter a headline, article, or factual claim to check.")
    return build_graph().invoke({"article": article, "retries": 0, "log": []})
