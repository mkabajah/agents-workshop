"""SESSION 1 · STEP 6 · LangGraph: when you KNOW the process, draw it. Put an agent in one node.

What you learn
    Not everything should be an agent. When the steps are known, write them as a graph: plain code
    for the easy steps, an agent only where judgement is needed, and a person where approval is needed.
    LangGraph gives you the pieces: a shared state, nodes (steps), edges (arrows), interrupt() (pause).

How to run
    python session1/06_langgraph_screening.py backend-engineer
    python session1/06_langgraph_screening.py ai-engineer --auto           # approve automatically (tests)
    MODEL=fake python session1/06_langgraph_screening.py backend-engineer   # offline, no API key

    START → load_job → source ──(nobody)──────────────────→ END
                         └──(candidates)→ screen → approve → save → END
                                         (agent)  (human)

  load_job   read the job post, pull out the must-haves               plain code
  source     find_candidates once per must-have, rank by matches       plain code: cheap, testable
  screen     an AGENT reads the top CVs and checks the evidence        the one step that needs judgement
  approve    interrupt(): a hiring manager approves or rejects         a human
  save       save_shortlist                                            plain code

What you'll see
    The shortlist, which must-haves each CV met, the screener's report, an [approve / reject]
    prompt, then "Done" and the graph as Mermaid text (paste it into mermaid.live to draw it).

Comes after steps 2-5: create_agent was itself a tiny graph (model <-> tools). Here you draw your own.

New to async/await, "async with" or decorators? Read docs/python-notes.md (5 minutes).
"""
import asyncio
import json
import pathlib
import sys
import uuid
from typing import TypedDict

# Let Python find the shared "workshop" package in the repo root (docs/python-notes.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from workshop import actions
from workshop.mcp_tools import rag_tools
from workshop.jobs import must_haves
from workshop.models import get_model
from workshop.prompts import SCREENER
from workshop.show import banner, content_text


# The STATE: a shared notebook. Every node (step) reads it and returns a few fields to update.
# TypedDict just says which fields exist and their types; total=False = none is required.
class Screening(TypedDict, total=False):
    job: str
    must_haves: list[str]          # e.g. ["java", "kafka", "docker"]
    matches: dict[str, list[str]]  # cv → must-haves found
    shortlist: list[str]
    report: str
    approved: bool


def as_json(result):
    """MCP tools return content blocks; turn them back into Python data."""
    if isinstance(result, list) and result and isinstance(result[0], dict) and "text" in result[0]:
        return [json.loads(b["text"]) for b in result]
    return json.loads(result) if isinstance(result, str) else result


def build(tools):
    """Wire the five steps into a LangGraph graph and return it, ready to run."""
    t = {tool.name: tool for tool in tools}                       # tools by name, e.g. t["get_cv"]
    # The only agent in the workflow: it may read CVs (get_cv, search_cvs) and nothing else.
    screener = create_agent(get_model(), [t["get_cv"], t["search_cvs"]], system_prompt=SCREENER)

    # Each node gets the current state `s` and returns ONLY the fields it changes;
    # LangGraph merges them into the state for the next node.
    async def load_job(s: Screening):
        """Read the job post and pull out its must-have skills."""
        jobs = as_json(await t["list_jobs"].ainvoke({}))
        post = next(j["text"] for j in jobs if j["job"] == s["job"])
        return {"must_haves": must_haves(post)}

    async def source(s: Screening):
        """Find every CV with each must-have and keep the 3 that match the most."""
        matches = {}
        for skill in s["must_haves"]:
            for hit in as_json(await t["find_candidates"].ainvoke({"skill": skill, "city": ""})):
                matches.setdefault(hit["cv"], []).append(skill)
        ranked = sorted(matches, key=lambda cv: -len(matches[cv]))
        return {"matches": matches, "shortlist": ranked[:3]}

    async def screen(s: Screening):
        """Ask the screener agent to check the evidence in the shortlisted CVs."""
        task = (f"Job: {s['job']}. Must-haves: {', '.join(s['must_haves'])}.\n"
                f"Check the evidence in these CVs: {', '.join(s['shortlist'])}.")
        out = await screener.ainvoke({"messages": [{"role": "user", "content": task}]})
        return {"report": content_text(out["messages"][-1])}

    def approve(s: Screening):
        """Pause for a hiring manager; continue with their answer."""
        # interrupt() PAUSES the whole graph here. The run returns to our code with "__interrupt__";
        # when we call it again with Command(resume="approve"), interrupt() returns "approve".
        decision = interrupt({"job": s["job"], "shortlist": s["shortlist"], "report": s["report"]})
        return {"approved": decision == "approve"}

    async def save(s: Screening):
        """Save the shortlist to outbox/ if it was approved."""
        if s.get("approved"):
            reasons = [f"has {', '.join(s['matches'][cv])}" for cv in s["shortlist"]]
            await actions.save_shortlist.ainvoke({"job": s["job"], "candidates": s["shortlist"], "reasons": reasons})
        return {}

    # Build the graph: nodes are the step functions above, edges are the arrows between them.
    g = StateGraph(Screening)
    for node in (load_job, source, screen, approve, save):           # node name = function name
        g.add_node(node)
    g.add_edge(START, "load_job")
    g.add_edge("load_job", "source")
    # A fork: after "source", go to "screen" if someone matched, otherwise straight to END.
    g.add_conditional_edges("source", lambda s: "screen" if s["shortlist"] else END, ["screen", END])
    g.add_edge("screen", "approve")
    g.add_edge("approve", "save")
    g.add_edge("save", END)
    return g.compile(checkpointer=InMemorySaver())                # a checkpointer is required to pause


async def run(job="backend-engineer", auto=False):
    """Screen candidates for one job: run until the approval pause, ask, then finish."""
    # A thread_id is needed because the graph pauses: the checkpointer saves the run under this id.
    cfg = {"configurable": {"thread_id": f"screen-{uuid.uuid4().hex[:6]}"}}
    # Start the CV MCP server once. "async with" stops it again at the end, even after an error.
    async with rag_tools() as tools:
        app = build(tools)
        # The first ainvoke runs load_job, source and screen, then stops at interrupt() in approve.
        result = await app.ainvoke({"job": job}, cfg)
        if result.get("__interrupt__"):
            pending = result["__interrupt__"][0].value
            banner(f"⏸  Approve the shortlist for {pending['job']}?")
            print("Shortlist:", pending["shortlist"])
            print("Must-haves met:", {cv: result["matches"][cv] for cv in pending["shortlist"]})
            print("Screener report:\n", pending["report"][:1500])
            answer = "approve" if auto else input("\n[approve / reject] ").strip()
            # Resume the SAME thread: interrupt() now returns `answer`, and approve -> save run.
            result = await app.ainvoke(Command(resume=answer), cfg)
    banner(f"Done · approved={result.get('approved')} · must-haves={result.get('must_haves')}")
    print(app.get_graph().draw_mermaid())                  # paste into mermaid.live to see the graph
    return result


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    # asyncio.run() starts the async function and waits until it is finished.
    asyncio.run(run(args[0] if args else "backend-engineer", "--auto" in sys.argv))
