"""
SESSION 1 · STEP 6 · LangGraph: when you KNOW the process, draw it. Put an agent in one node.

    python session1/06_langgraph_screening.py backend-engineer
    python session1/06_langgraph_screening.py ai-engineer --auto

    START → load_job → source ──(nobody)──────────────────→ END
                         └──(candidates)→ screen → approve → save → END
                                         (agent)  (human)

  load_job   read the job post, pull out the must-haves               plain code
  source     find_candidates once per must-have, rank by matches       plain code: cheap, testable
  screen     an AGENT reads the top CVs and checks the evidence        the one step that needs judgement
  approve    interrupt(): a hiring manager approves or rejects         a human
  save       save_shortlist                                            plain code
"""
import asyncio
import json
import pathlib
import re
import sys
import uuid
from typing import TypedDict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from workshop import actions
from workshop.mcp_tools import rag_tools
from workshop.models import SKILLS, get_model
from workshop.prompts import SCREENER
from workshop.show import banner, content_text


class Screening(TypedDict, total=False):
    job: str
    must_haves: list[str]          # e.g. ["java", "kafka", "docker"]
    matches: dict[str, list[str]]  # cv → must-haves found
    shortlist: list[str]
    report: str
    approved: bool


def must_haves(post: str) -> list[str]:
    """The skills named under '## Must have' (first known skill on each line)."""
    block = post.split("## Must have", 1)[-1].split("##", 1)[0].lower()
    out = []
    for line in block.splitlines():
        hits = sorted((m.start(), s) for s in SKILLS for m in [re.search(rf"(?<![a-z]){re.escape(s)}(?![a-z])", line)] if m)
        if hits:
            out.append(hits[0][1])
    return out


def as_json(result):
    """MCP tools return content blocks; turn them back into Python data."""
    if isinstance(result, list) and result and isinstance(result[0], dict) and "text" in result[0]:
        return [json.loads(b["text"]) for b in result]
    return json.loads(result) if isinstance(result, str) else result


def build(tools):
    t = {tool.name: tool for tool in tools}
    screener = create_agent(get_model(), [t["get_cv"], t["search_cvs"]], system_prompt=SCREENER)

    async def load_job(s: Screening):
        jobs = as_json(await t["list_jobs"].ainvoke({}))
        post = next(j["text"] for j in jobs if j["job"] == s["job"])
        return {"must_haves": must_haves(post)}

    async def source(s: Screening):
        matches = {}
        for skill in s["must_haves"]:
            for hit in as_json(await t["find_candidates"].ainvoke({"skill": skill, "city": ""})):
                matches.setdefault(hit["cv"], []).append(skill)
        ranked = sorted(matches, key=lambda cv: -len(matches[cv]))
        return {"matches": matches, "shortlist": ranked[:3]}

    async def screen(s: Screening):
        task = (f"Job: {s['job']}. Must-haves: {', '.join(s['must_haves'])}.\n"
                f"Check the evidence in these CVs: {', '.join(s['shortlist'])}.")
        out = await screener.ainvoke({"messages": [{"role": "user", "content": task}]})
        return {"report": content_text(out["messages"][-1])}

    def approve(s: Screening):
        decision = interrupt({"job": s["job"], "shortlist": s["shortlist"], "report": s["report"]})
        return {"approved": decision == "approve"}

    async def save(s: Screening):
        if s.get("approved"):
            reasons = [f"has {', '.join(s['matches'][cv])}" for cv in s["shortlist"]]
            await actions.save_shortlist.ainvoke({"job": s["job"], "candidates": s["shortlist"], "reasons": reasons})
        return {}

    g = StateGraph(Screening)
    for node in (load_job, source, screen, approve, save):
        g.add_node(node)
    g.add_edge(START, "load_job")
    g.add_edge("load_job", "source")
    g.add_conditional_edges("source", lambda s: "screen" if s["shortlist"] else END, ["screen", END])
    g.add_edge("screen", "approve")
    g.add_edge("approve", "save")
    g.add_edge("save", END)
    return g.compile(checkpointer=InMemorySaver())


async def run(job="backend-engineer", auto=False):
    cfg = {"configurable": {"thread_id": f"screen-{uuid.uuid4().hex[:6]}"}}
    async with rag_tools() as tools:
        app = build(tools)
        result = await app.ainvoke({"job": job}, cfg)
        if result.get("__interrupt__"):
            pending = result["__interrupt__"][0].value
            banner(f"⏸  Approve the shortlist for {pending['job']}?")
            print("Shortlist:", pending["shortlist"])
            print("Must-haves met:", {cv: result["matches"][cv] for cv in pending["shortlist"]})
            print("Screener report:\n", pending["report"][:1500])
            answer = "approve" if auto else input("\n[approve / reject] ").strip()
            result = await app.ainvoke(Command(resume=answer), cfg)
    banner(f"Done · approved={result.get('approved')} · must-haves={result.get('must_haves')}")
    print(app.get_graph().draw_mermaid())                  # paste into mermaid.live to see the graph
    return result


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    asyncio.run(run(args[0] if args else "backend-engineer", "--auto" in sys.argv))
