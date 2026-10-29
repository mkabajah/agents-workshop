"""
STRANDS · STEP 6 · A workflow you can draw, with an agent in one step and a person in another.

What it does
    Screens candidates for one job in five fixed steps. Only ONE step needs a model:

    load_job (code) → source (code) → [nobody? stop] → screen (AGENT) → approve (PERSON) → save (code)

How to run
    python run.py 06 backend-engineer --framework strands
    python frameworks/strands/06_workflow.py backend-engineer
    python frameworks/strands/06_workflow.py backend-engineer --auto      # approve automatically (tests)

What you'll see
    The must-haves and the shortlist (for backend-engineer: cand-006, cand-018, cand-007), the
    screener's report, a pause for your approval, and "Done · approved = True". An approved shortlist
    is written to outbox/shortlists.jsonl.

Same thing in LangChain: session1/06_langgraph_screening.py (a LangGraph StateGraph with interrupt()).

Why plain Python here
    Strands is model-driven: an Agent decides its own steps. For a FIXED process the simplest Strands
    answer is plain Python for the steps you know, and an Agent only where judgement is needed.
    (For graphs whose nodes are agents, Strands has GraphBuilder: see examples/09_strands_graph.py.)
"""
import json
import pathlib
import sys

# Let Python find the shared "workshop" package that lives two folders up (the repo root).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from strands import Agent

from workshop.jobs import must_haves
from workshop.prompts import SCREENER
from workshop.strands_kit import cv_client, save_shortlist, strands_model


def run(job="backend-engineer", auto=False):
    """Run the five steps for one job and return whether the shortlist was approved."""
    with cv_client() as cv:
        # 1 · load_job (code): read the job post and its must-haves.
        # call_tool_sync(id, tool, args) calls one MCP tool directly, no model involved.
        # Each result item is a text block holding JSON, so we json.loads() it.
        jobs = [json.loads(c["text"]) for c in cv.call_tool_sync("wf-jobs", "list_jobs", {})["content"]]
        post = next(j["text"] for j in jobs if j["job"] == job)
        skills = must_haves(post)

        # 2 · source (code): who has each must-have? rank by how many they have
        matches = {}                                                       # CV file name -> skills it has
        for skill in skills:
            for c in cv.call_tool_sync(f"wf-{skill}", "find_candidates", {"skill": skill, "city": ""})["content"]:
                matches.setdefault(json.loads(c["text"])["cv"], []).append(skill)
        shortlist = sorted(matches, key=lambda k: -len(matches[k]))[:3]    # most must-haves first, top 3
        print("must-haves:", skills, "· shortlist:", shortlist)
        if not shortlist:                                                  # the fork in the road
            print("Nobody matches: stop here. No agent ran, no tokens spent.")
            return {"approved": False, "shortlist": []}

        # 3 · screen (AGENT): the one step that needs judgement.
        # The screener only gets the two read tools it needs: least privilege.
        tools = [t for t in cv.list_tools_sync() if t.tool_name in ("get_cv", "search_cvs")]
        screener = Agent(model=strands_model(), tools=tools, system_prompt=SCREENER, callback_handler=None)
        report = str(screener(f"Job: {job}. Must-haves: {', '.join(skills)}. Check the evidence in: {', '.join(shortlist)}."))

        # 4 · approve (PERSON)
        print(f"\n⏸  Approve the shortlist for {job}? {shortlist}\nScreener report:\n{report[:1200]}")
        approved = auto or input("[approve / reject] ").strip() == "approve"

        # 5 · save (code): a Strands @tool is still a normal Python function, so we can call it directly.
        if approved:
            save_shortlist(job=job, candidates=shortlist, reasons=[f"has {', '.join(matches[c])}" for c in shortlist])
        print("Done · approved =", approved)
        return {"approved": approved, "shortlist": shortlist}


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(args[0] if args else "backend-engineer", "--auto" in sys.argv)
