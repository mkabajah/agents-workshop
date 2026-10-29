"""
EXAMPLE 04 · Router: one classification, then a fixed branch.      Pattern: router (LangGraph)

    python examples/04_router.py "Which candidates know Python?"        → LIST   → find_candidates
    python examples/04_router.py "What did cand-004 build at Orbit?"    → SEARCH → search_cvs
    python examples/04_router.py "What are the backend must-haves?"     → JOB    → list_jobs

What it shows: the model decides ONCE (which kind of question is this?), then plain code runs the right
tool. No loop, so it is cheaper, faster and more predictable than an agent.

                    ┌─ LIST   ─▶ list_all  (find_candidates) ─┐
    START ─▶ classify ─ SEARCH ─▶ search    (search_cvs)      ─┼─▶ END
                    └─ JOB    ─▶ jobs      (list_jobs)       ─┘

When you'd use it: a help desk or search box where the kinds of questions are known in advance.
What you'll see: the chosen route and the tool's result.
With MODEL=fake the classifier falls back to keywords.

New to async/await, "async with" or decorators? Read docs/python-notes.md (5 minutes).
"""
import asyncio
import pathlib
import sys
from typing import Literal, TypedDict

# Let Python find the shared "workshop" package in the repo root (docs/python-notes.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from workshop import config
from workshop.mcp_tools import rag_tools
from workshop.models import SKILLS, get_model


class Route(BaseModel):
    """The classifier's decision. Literal[...] means the model may ONLY answer one of these three words."""

    kind: Literal["LIST", "SEARCH", "JOB"]
    term: str


class S(TypedDict, total=False):
    """The graph's state: the question, the route chosen, and the tool result."""

    question: str
    route: Route
    result: str


# async def: this function waits on the model and the MCP server; each wait is marked with await.
async def classify(s: S) -> dict:
    """Decide which kind of question this is (one model call, or keywords with the fake model)."""
    q = s["question"]
    if config.MODEL != "fake":
        # with_structured_output(Route) = answer with a Route object, not free text, so code can branch on it.
        router = get_model().with_structured_output(Route)
        return {"route": await router.ainvoke(
            "Classify the recruiter question. LIST = 'which/who/how many candidates have X' (term = X). "
            "JOB = about a job post. SEARCH = anything else (term = the question).\n\n" + q)}
    low = q.lower()
    if "job" in low or "must-have" in low:
        return {"route": Route(kind="JOB", term="")}
    skill = next((k for k in SKILLS if k in low), "")
    if skill and any(w in low for w in ("which", "who", "how many")):
        return {"route": Route(kind="LIST", term=skill)}
    return {"route": Route(kind="SEARCH", term=q)}


def build(tools):
    """Wire the graph: classify first, then exactly one of three tool nodes."""
    t = {x.name: x for x in tools}

    async def list_all(s: S):
        return {"result": str(await t["find_candidates"].ainvoke({"skill": s["route"].term, "city": ""}))}

    async def search(s: S):
        return {"result": str(await t["search_cvs"].ainvoke({"question": s["route"].term}))}

    async def jobs(s: S):
        return {"result": str(await t["list_jobs"].ainvoke({}))}

    g = StateGraph(S)
    g.add_node("classify", classify)
    g.add_node("list_all", list_all)
    g.add_node("search", search)
    g.add_node("jobs", jobs)
    g.add_edge(START, "classify")
    # The router itself: read the route from the state and map it to the name of the next node.
    g.add_conditional_edges("classify", lambda s: {"LIST": "list_all", "SEARCH": "search", "JOB": "jobs"}[s["route"].kind],
                            ["list_all", "search", "jobs"])
    for n in ("list_all", "search", "jobs"):
        g.add_edge(n, END)
    return g.compile()


async def run(question: str):
    """Start the CV server, run the graph once, print the route and the result."""
    # Start the CV MCP server once. "async with" stops it again at the end, even after an error.
    async with rag_tools() as tools:
        out = await build(tools).ainvoke({"question": question})
    print(f"route = {out['route'].kind} ({out['route'].term!r})\n{out['result'][:600]}")
    return out


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    # asyncio.run() starts the async function and waits until it is finished.
    asyncio.run(run(" ".join(sys.argv[1:]) or "Which candidates know Python?"))
