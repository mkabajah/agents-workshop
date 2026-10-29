"""
EXAMPLE 07 · Fan-out in parallel, then merge.        Pattern: map-reduce with LangGraph Send

    python examples/07_parallel_fanout.py backend-engineer

What it shows: one branch per must-have runs AT THE SAME TIME, each calling find_candidates; their
results are merged and candidates are ranked by how many must-haves they meet.

                        ┌─▶ search_skill(java)   ─┐
    START ─▶ read_job ──┼─▶ search_skill(kafka)  ─┼─▶ merge (rank) ─▶ END
                        └─▶ search_skill(docker) ─┘

Two LangGraph ideas:
  Send     "start this node once per item, with this input": the number of branches is decided at run time.
  reducer  Annotated[list, operator.add] on a state field = "when several branches write here, ADD the
           lists together" instead of the last one overwriting the others.

When you'd use it: many independent lookups (one per skill, per city, per document) where waiting for
each in turn would be slow. No model needed here, so it is fast and testable: agents only where
judgement is needed.
What you'll see: the must-haves, the time taken, and the top candidates with the skills they meet.

New to async/await, "async with" or decorators? Read docs/python-notes.md (5 minutes).
"""
import asyncio
import json
import operator
import pathlib
import sys
import time
from typing import Annotated, TypedDict

# Let Python find the shared "workshop" package in the repo root (docs/python-notes.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from workshop.mcp_tools import rag_tools
from workshop.models import SKILLS
from workshop.show import banner


class S(TypedDict, total=False):
    """The graph's state. 'hits' is the field every parallel branch writes into."""

    job: str
    must_haves: list[str]
    hits: Annotated[list[tuple[str, str]], operator.add]     # reducer: branches' results are concatenated
    ranking: list[tuple[str, list[str]]]


class Branch(TypedDict):
    """The small input each parallel branch gets: just its skill."""

    skill: str


def parse(result):
    """MCP tools return content blocks of JSON text; turn them back into Python dicts and lists."""
    if isinstance(result, list) and result and isinstance(result[0], dict) and "text" in result[0]:
        return [json.loads(b["text"]) for b in result]
    return json.loads(result) if isinstance(result, str) else result


def build(tools):
    """Wire the graph: read the job, fan out one branch per must-have, merge and rank."""
    t = {x.name: x for x in tools}

    async def read_job(s: S):
        post = next(j["text"] for j in parse(await t["list_jobs"].ainvoke({})) if j["job"] == s["job"])
        block = post.split("## Must have", 1)[-1].split("##", 1)[0].lower()
        return {"must_haves": [k for k in SKILLS if k in block][:6]}

    # fan_out returns a LIST of Send objects instead of one node name: LangGraph runs them all in parallel.
    def fan_out(s: S):
        return [Send("search_skill", {"skill": k}) for k in s["must_haves"]]   # one branch per skill

    async def search_skill(b: Branch):
        found = parse(await t["find_candidates"].ainvoke({"skill": b["skill"], "city": ""}))
        return {"hits": [(h["cv"], b["skill"]) for h in found]}

    # merge runs once, after ALL branches finished; by then the reducer has collected every branch's hits.
    def merge(s: S):
        by_cv = {}
        for cv, skill in s["hits"]:
            by_cv.setdefault(cv, []).append(skill)
        return {"ranking": sorted(by_cv.items(), key=lambda kv: -len(kv[1]))}

    g = StateGraph(S)
    g.add_node("read_job", read_job)
    g.add_node("search_skill", search_skill)
    g.add_node("merge", merge)
    g.add_edge(START, "read_job")
    g.add_conditional_edges("read_job", fan_out, ["search_skill"])
    g.add_edge("search_skill", "merge")
    g.add_edge("merge", END)
    return g.compile()


# async def: this function waits on the model and the MCP server; each wait is marked with await.
async def run(job="backend-engineer"):
    """Run the fan-out for one job and print the top 5 candidates."""
    # Start the CV MCP server once. "async with" stops it again at the end, even after an error.
    async with rag_tools() as tools:
        start = time.perf_counter()
        out = await build(tools).ainvoke({"job": job})
    banner(f"{job}: {len(out['must_haves'])} branches in parallel · {time.perf_counter() - start:.1f}s")
    print("must-haves:", out["must_haves"])
    for cv, skills in out["ranking"][:5]:
        print(f"  {cv:<18} {len(skills)}/{len(out['must_haves'])}  {', '.join(skills)}")
    return out


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    # asyncio.run() starts the async function and waits until it is finished.
    asyncio.run(run(sys.argv[1] if len(sys.argv) > 1 else "backend-engineer"))
