"""
SESSION 2 · STEP 7 · Multi-agent with LangChain: a lead agent whose tools are other agents.

    python session2/07_supervisor.py "Shortlist 3 candidates for the backend job and book an interview with the best one"
    python session2/07_supervisor.py --auto ...      # approve every delegation to the coordinator

    lead ── sourcer      (find_candidates, search_cvs)     who has each must-have?
         ├─ screener     (get_cv, search_cvs)               is the evidence real? risks?
         └─ coordinator  (get_free_slots, schedule_interview, send_email)   ← human approves

Why split: each specialist sees only ITS tools and ITS context. The raw CV text the screener
reads never reaches the lead: only the screener's short report does.

New to async/await, "async with" or decorators? Read PYTHON_NOTES.md (5 minutes).
"""
import argparse
import asyncio
import pathlib
import sys
import uuid

# Let Python find the shared "workshop" package in the repo root (PYTHON_NOTES.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ModelCallLimitMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from workshop import actions, prompts
from workshop.mcp_tools import rag_tools
from workshop.models import get_model
from workshop.show import banner, content_text, trace


def build(tools):
    t = {x.name: x for x in tools}
    limit = [ModelCallLimitMiddleware(run_limit=10)]
    sourcer = create_agent(get_model(), [t["find_candidates"], t["search_cvs"]], system_prompt=prompts.SOURCER, middleware=limit)
    screener = create_agent(get_model(), [t["get_cv"], t["search_cvs"]], system_prompt=prompts.SCREENER, middleware=limit)
    coordinator = create_agent(get_model(), [actions.get_free_slots, actions.schedule_interview, actions.send_email],
                               system_prompt=prompts.COORDINATOR, middleware=limit)

    async def delegate(agent, task: str) -> str:
        out = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
        return content_text(out["messages"][-1])            # only the final report goes back to the lead

    @tool("sourcer")
    async def call_sourcer(task: str) -> str:
        """Find EVERY candidate who has given skills or must-haves. Give the must-haves explicitly."""
        return await delegate(sourcer, task)

    @tool("screener")
    async def call_screener(task: str) -> str:
        """Check the evidence in specific CVs against must-haves. Give CV file names and must-haves."""
        return await delegate(screener, task)

    @tool("coordinator")
    async def call_coordinator(task: str) -> str:
        """Book an interview or email ONE named candidate (CV file name, job id). A human approves first."""
        return await delegate(coordinator, task)

    return create_agent(
        get_model(),
        tools=[t["list_jobs"], call_sourcer, call_screener, call_coordinator],
        system_prompt=prompts.LEAD,
        checkpointer=InMemorySaver(),
        middleware=[ModelCallLimitMiddleware(run_limit=12),
                    HumanInTheLoopMiddleware(interrupt_on={"coordinator": True})],   # approve the delegation
    )


# async def: this function waits on the model and the MCP server; each wait is marked with await.
async def run(question: str, auto=False):
    cfg = {"configurable": {"thread_id": f"lead-{uuid.uuid4().hex[:6]}"}}
    # Start the CV MCP server once. "async with" stops it again at the end, even after an error.
    async with rag_tools() as tools:
        lead = build(tools)
        result = await lead.ainvoke({"messages": [{"role": "user", "content": question}]}, cfg)
        while result.get("__interrupt__"):
            decisions = []
            for action in result["__interrupt__"][0].value["action_requests"]:
                print(f"\n⏸  lead → coordinator: {action['args'].get('task')}")
                ok = auto or input("   approve? [y/n] ").strip().lower().startswith("y")
                decisions.append({"type": "approve"} if ok else {"type": "reject", "message": "Not approved by the hiring manager."})
            result = await lead.ainvoke(Command(resume={"decisions": decisions}), cfg)
    banner("What the LEAD saw (the specialists' own steps stay inside them)")
    trace(result["messages"])
    return result


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="*")
    p.add_argument("--auto", action="store_true")
    a = p.parse_args()
    # asyncio.run() starts the async function and waits until it is finished.
    asyncio.run(run(" ".join(a.question) or
                    "Shortlist 3 candidates for the backend job and book an interview with the best one.", a.auto))
