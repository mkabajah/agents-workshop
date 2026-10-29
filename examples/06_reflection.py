"""
EXAMPLE 06 · Reflection: a writer and a critic loop until it's good.   Pattern: evaluator–optimizer

    python examples/06_reflection.py cand-006.md backend-engineer

The writer drafts an interview invitation from the CV evidence; the critic checks it against a
rubric (specific, fair, short, no promises, no personal data) and returns PASS or fixes.
At most 3 rounds: a loop always needs a budget. Built as a 2-node LangGraph with a cycle.

New to async/await, "async with" or decorators? Read PYTHON_NOTES.md (5 minutes).
"""
import asyncio
import pathlib
import sys
from typing import TypedDict

# Let Python find the shared "workshop" package in the repo root (PYTHON_NOTES.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langgraph.graph import END, START, StateGraph

from workshop import config
from workshop.mcp_tools import rag_tools
from workshop.models import get_model
from workshop.show import banner, content_text

RUBRIC = ("1) mentions one specific skill from the CV, 2) neutral and fair, 3) under 120 words, "
          "4) promises nothing about salary or the outcome, 5) no phone numbers or emails")
MAX_ROUNDS = 3


class S(TypedDict, total=False):
    cv_text: str
    job: str
    draft: str
    feedback: str
    rounds: int


# async def: this function waits on the model and the MCP server; each wait is marked with await.
async def write(s: S):
    prompt = (f"Write a short interview invitation for the {s['job']} role, based on this CV:\n{s['cv_text'][:2500]}\n"
              + (f"\nRevise using this feedback: {s['feedback']}\nPrevious draft:\n{s['draft']}" if s.get("feedback") else ""))
    reply = await get_model().ainvoke(prompt)
    return {"draft": content_text(reply), "rounds": s.get("rounds", 0) + 1}


async def critique(s: S):
    if config.MODEL == "fake":                                     # the scripted model can't judge: pass after 2 rounds
        return {"feedback": "PASS" if s["rounds"] >= 2 else "Name one specific skill from the CV."}
    reply = await get_model().ainvoke(f"Check this email against the rubric: {RUBRIC}.\nReply PASS, or list the "
                                      f"fixes in one line.\n\nEMAIL:\n{s['draft']}")
    return {"feedback": content_text(reply).strip()}


def route(s: S):
    done = s["feedback"].upper().startswith("PASS") or s["rounds"] >= MAX_ROUNDS
    return END if done else "write"


async def run(cv="cand-006.md", job="backend-engineer"):
    # Start the CV MCP server once. "async with" stops it again at the end, even after an error.
    async with rag_tools() as tools:
        get_cv = next(t for t in tools if t.name == "get_cv")
        cv_text = str(await get_cv.ainvoke({"file_name": cv}))
    g = StateGraph(S)
    g.add_node("write", write)
    g.add_node("critique", critique)
    g.add_edge(START, "write")
    g.add_edge("write", "critique")
    g.add_conditional_edges("critique", route, ["write", END])
    out = await g.compile().ainvoke({"cv_text": cv_text, "job": job})
    banner(f"After {out['rounds']} round(s) · critic: {out['feedback'][:80]}")
    print(out["draft"])
    return out


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    a = sys.argv[1:]
    # asyncio.run() starts the async function and waits until it is finished.
    asyncio.run(run(*(a[:2] if len(a) >= 2 else ["cand-006.md", "backend-engineer"])))
