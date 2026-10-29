"""
STRANDS · STEP 6 · A workflow you can draw, with an agent in one step and a person in another.
Same process as session1/06_langgraph_screening.py (LangGraph StateGraph).

    python frameworks/strands/06_workflow.py backend-engineer
    python frameworks/strands/06_workflow.py backend-engineer --auto      # approve automatically (tests)

Strands is model-driven, so for a fixed process the simplest Strands answer is: plain Python for
the steps you know, an Agent where judgement is needed. (For graphs of AGENTS, Strands has
GraphBuilder: see examples/09_strands_graph.py.)

    load_job (code) → source (code) → [nobody? stop] → screen (AGENT) → approve (PERSON) → save (code)
"""
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from strands import Agent

from workshop.models import SKILLS
from workshop.prompts import SCREENER
from workshop.strands_kit import cv_client, save_shortlist, strands_model


def must_haves(post: str) -> list[str]:
    """The skills named under '## Must have' (first known skill on each line), as in the LangGraph version."""
    block = post.split("## Must have", 1)[-1].split("##", 1)[0].lower()
    out = []
    for line in block.splitlines():
        hits = sorted((m.start(), s) for s in SKILLS for m in [re.search(rf"(?<![a-z]){re.escape(s)}(?![a-z])", line)] if m)
        if hits:
            out.append(hits[0][1])
    return out


def run(job="backend-engineer", auto=False):
    with cv_client() as cv:
        # 1 · load_job (code): read the job post and its must-haves
        jobs = [json.loads(c["text"]) for c in cv.call_tool_sync("wf-jobs", "list_jobs", {})["content"]]
        post = next(j["text"] for j in jobs if j["job"] == job)
        skills = must_haves(post)

        # 2 · source (code): who has each must-have? rank by how many they have
        matches = {}
        for skill in skills:
            for c in cv.call_tool_sync(f"wf-{skill}", "find_candidates", {"skill": skill, "city": ""})["content"]:
                matches.setdefault(json.loads(c["text"])["cv"], []).append(skill)
        shortlist = sorted(matches, key=lambda k: -len(matches[k]))[:3]
        print("must-haves:", skills, "· shortlist:", shortlist)
        if not shortlist:                                                  # the fork in the road
            print("Nobody matches: stop here. No agent ran, no tokens spent.")
            return {"approved": False, "shortlist": []}

        # 3 · screen (AGENT): the one step that needs judgement
        tools = [t for t in cv.list_tools_sync() if t.tool_name in ("get_cv", "search_cvs")]
        screener = Agent(model=strands_model(), tools=tools, system_prompt=SCREENER, callback_handler=None)
        report = str(screener(f"Job: {job}. Must-haves: {', '.join(skills)}. Check the evidence in: {', '.join(shortlist)}."))

        # 4 · approve (PERSON)
        print(f"\n⏸  Approve the shortlist for {job}? {shortlist}\nScreener report:\n{report[:1200]}")
        approved = auto or input("[approve / reject] ").strip() == "approve"

        # 5 · save (code)
        if approved:
            save_shortlist(job=job, candidates=shortlist, reasons=[f"has {', '.join(matches[c])}" for c in shortlist])
        print("Done · approved =", approved)
        return {"approved": approved, "shortlist": shortlist}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(args[0] if args else "backend-engineer", "--auto" in sys.argv)
