"""SESSION 1 · STEP 4 · Middleware: limits, summarisation and HUMAN APPROVAL before any write.

What you learn
    Middleware = rules that run at fixed moments of the agent loop (before/after a model call,
    around a tool call), without changing the agent itself. Here: a summary of long chats, a limit on
    model calls, a limit on get_cv, and a pause for a person before every write tool.

How to run
    python session1/04_middleware_hitl.py "Find a Kafka candidate and book an interview for the backend job"
    python session1/04_middleware_hitl.py --auto ...                    # approve everything (tests)
    python session1/04_middleware_hitl.py --sqlite hire.db ...          # pause, press Ctrl+C, then later:
    python session1/04_middleware_hitl.py --sqlite hire.db --resume <thread-id>
    MODEL=fake python session1/04_middleware_hitl.py --auto             # offline, no API key

What you'll see
    The thread_id, then a pause like "The agent wants to run schedule_interview({...})" and a prompt:
    [a]pprove, [e]dit the arguments, or [r]eject with a reason. Afterwards, the trace, and one new
    line in outbox/interviews.jsonl if you approved. Rejected = nothing written; the model reads why.

How it works
    The middleware stops the graph BEFORE the tool runs and saves the state in the checkpointer.
    ainvoke returns early with "__interrupt__". We ask a person, then call ainvoke again with
    Command(resume=...) on the same thread_id, and the run continues exactly where it stopped.

Comes after step 3: the checkpointer from step 3 is what makes the pause possible. Step 5 shows how
to write your own middleware.

New to async/await, "async with" or decorators? Read docs/python-notes.md (5 minutes).
"""
import argparse
import asyncio
import json
import pathlib
import sys
import uuid

# Let Python find the shared "workshop" package in the repo root (docs/python-notes.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.agents.middleware import (HumanInTheLoopMiddleware, ModelCallLimitMiddleware,
                                         SummarizationMiddleware, ToolCallLimitMiddleware)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from workshop import actions, config
from workshop.mcp_tools import rag_tools
from workshop.models import get_model
from workshop.prompts import HIRING_AGENT
from workshop.show import banner, trace


def build_agent(tools, checkpointer):
    """The hiring agent with read + write tools and four middleware rules.

    Middleware runs in list order on the way in to the model and in reverse order on the way out,
    like layers of an onion. The approval rule is last so it sees the final tool call."""
    return create_agent(
        model=get_model(),
        tools=tools + actions.ALL,
        system_prompt=HIRING_AGENT,
        checkpointer=checkpointer,                                   # required: a pause must be saved
        middleware=[
            SummarizationMiddleware(model=get_model(config.SMALL_MODEL),  # long chats: compress old turns
                                    trigger=("tokens", 6000), keep=("messages", 20)),
            ModelCallLimitMiddleware(run_limit=12),                  # runaway loops stop here
            ToolCallLimitMiddleware(tool_name="get_cv", run_limit=6),  # reading every CV costs tokens
            HumanInTheLoopMiddleware(interrupt_on={                  # writes wait for a human
                # Per tool: which answers a person may give. True = approve/reject only.
                "schedule_interview": {"allowed_decisions": ["approve", "edit", "reject"]},
                "send_email": {"allowed_decisions": ["approve", "edit", "reject"]},
                "save_shortlist": True,
                "search_cvs": False,                                 # reads: no approval
            }),
        ],
    )


def ask_human(request: dict, auto: bool) -> list[dict]:
    """Ask a person about each paused tool call and return one decision per call.

    In a product this is a button in Slack or your admin page. The decision shapes are fixed by
    LangChain: {"type": "approve"}, {"type": "edit", "edited_action": ...},
    {"type": "reject", "message": ...}."""
    decisions = []
    for action in request["action_requests"]:
        print(f"\n⏸  The agent wants to run {action['name']}({json.dumps(action['args'], ensure_ascii=False)})")
        choice = "a" if auto else input("   [a]pprove · [e]dit · [r]eject ? ").strip().lower()[:1]
        if choice == "e":
            args = json.loads(input("   new arguments as JSON: "))
            decisions.append({"type": "edit", "edited_action": {"name": action["name"], "args": args}})
        elif choice == "r":
            decisions.append({"type": "reject", "message": input("   why? ") or "Not approved."})
        else:
            decisions.append({"type": "approve"})
    return decisions


async def drive(agent, payload, cfg, auto):
    """Run until the agent finishes; every interrupt goes to a person, then we resume."""
    # The approval loop. When the middleware pauses the agent, ainvoke RETURNS early with "__interrupt__".
    # We ask a person, then call ainvoke again with Command(resume=...) on the SAME thread to continue.
    result = await agent.ainvoke(payload, cfg)
    while result.get("__interrupt__"):
        request = result["__interrupt__"][0].value                   # {"action_requests": [...], ...}
        result = await agent.ainvoke(Command(resume={"decisions": ask_human(request, auto)}), cfg)
    return result


async def run(question, auto=False, sqlite=None, resume=None):
    """Start (or resume) one hiring request on its own thread and print what happened."""
    # A new thread_id for each run, unless we resume an old one. It is the key the checkpointer uses.
    thread = resume or f"hire-{uuid.uuid4().hex[:6]}"
    cfg = {"configurable": {"thread_id": thread}}
    print(f"thread_id = {thread}   (Ctrl+C at a prompt, then --sqlite FILE --resume {thread})")
    # Start the CV MCP server once. "async with" stops it again at the end, even after an error.
    async with rag_tools() as tools:
        if sqlite:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            # A SQLite file as the checkpointer: a paused run survives Ctrl+C and can be resumed later.
            async with AsyncSqliteSaver.from_conn_string(sqlite) as saver:
                agent = build_agent(tools, saver)
                result = await _go(agent, question, cfg, auto, resume)
        else:
            agent = build_agent(tools, InMemorySaver())
            result = await _go(agent, question, cfg, auto, resume)
    banner(f"thread {thread} · done")
    trace(result["messages"])
    print(f"\nWhat really happened: see {config.OUTBOX}/")
    return result


async def _go(agent, question, cfg, auto, resume):
    """Pick the first input: a new question, or the human decision for a run paused earlier."""
    if resume:                                                       # continue a pause from an earlier run
        state = await agent.aget_state(cfg)                          # load the paused run from SQLite
        if not state.interrupts:
            sys.exit("Nothing to resume on that thread.")
        payload = Command(resume={"decisions": ask_human(state.interrupts[0].value, auto)})
    else:
        payload = {"messages": [{"role": "user", "content": question}]}
    return await drive(agent, payload, cfg, auto)


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    # argparse reads the command-line options (--auto, --sqlite, --resume) into variables.
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="*")
    p.add_argument("--auto", action="store_true", help="approve every action (for tests)")
    p.add_argument("--sqlite", help="a SQLite file for durable checkpoints, e.g. checkpoints.db")
    p.add_argument("--resume", help="a thread id to resume (with --sqlite)")
    a = p.parse_args()
    q = " ".join(a.question) or "Find a Kafka candidate and book an interview for the backend job."
    # asyncio.run() starts the async function and waits until it is finished.
    asyncio.run(run(q, a.auto, a.sqlite, a.resume))
