"""
EXAMPLE 07 · Fan-out in parallel, then merge.        Pattern: map-reduce with LangGraph Send

    python examples/07_parallel_fanout.py backend-engineer

One branch per must-have runs AT THE SAME TIME (Send), each calling find_candidates; a reducer
merges the lists; the merge node ranks candidates by how many must-haves they meet.
No model needed for this part, so it is fast and testable: agents only where judgement is needed.
"""
import asyncio
import json
import operator
import pathlib
import sys
import time
from typing import Annotated, TypedDict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from workshop.mcp_tools import rag_tools
from workshop.models import SKILLS
from workshop.show import banner


class S(TypedDict, total=False):
    job: str
    must_haves: list[str]
    hits: Annotated[list[tuple[str, str]], operator.add]     # reducer: branches' results are concatenated
    ranking: list[tuple[str, list[str]]]


class Branch(TypedDict):
    skill: str


def parse(result):
    if isinstance(result, list) and result and isinstance(result[0], dict) and "text" in result[0]:
        return [json.loads(b["text"]) for b in result]
    return json.loads(result) if isinstance(result, str) else result


def build(tools):
    t = {x.name: x for x in tools}

    async def read_job(s: S):
        post = next(j["text"] for j in parse(await t["list_jobs"].ainvoke({})) if j["job"] == s["job"])
        block = post.split("## Must have", 1)[-1].split("##", 1)[0].lower()
        return {"must_haves": [k for k in SKILLS if k in block][:6]}

    def fan_out(s: S):
        return [Send("search_skill", {"skill": k}) for k in s["must_haves"]]   # one branch per skill

    async def search_skill(b: Branch):
        found = parse(await t["find_candidates"].ainvoke({"skill": b["skill"], "city": ""}))
        return {"hits": [(h["cv"], b["skill"]) for h in found]}

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


async def run(job="backend-engineer"):
    async with rag_tools() as tools:
        start = time.perf_counter()
        out = await build(tools).ainvoke({"job": job})
    banner(f"{job}: {len(out['must_haves'])} branches in parallel · {time.perf_counter() - start:.1f}s")
    print("must-haves:", out["must_haves"])
    for cv, skills in out["ranking"][:5]:
        print(f"  {cv:<18} {len(skills)}/{len(out['must_haves'])}  {', '.join(skills)}")
    return out


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1] if len(sys.argv) > 1 else "backend-engineer"))
