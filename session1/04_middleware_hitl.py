"""
SESSION 1 · STEP 4 · Middleware: limits, summarisation and HUMAN APPROVAL before any write.

    python session1/04_middleware_hitl.py "Find a Kafka candidate and book an interview for the backend job"
    python session1/04_middleware_hitl.py --sqlite ...          # pause, quit, resume later:
    python session1/04_middleware_hitl.py --resume <thread-id>  #   (needs the same --sqlite file)
    python session1/04_middleware_hitl.py --auto ...            # approve everything (tests)

The agent now has action tools (schedule_interview, send_email). Reads run freely; every write
stops the graph and waits for a person: approve, edit the arguments, or reject with a reason.
Look in outbox/ afterwards to see what really "happened".
"""
import argparse
import asyncio
import json
import pathlib
import sys
import uuid

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
                "schedule_interview": {"allowed_decisions": ["approve", "edit", "reject"]},
                "send_email": {"allowed_decisions": ["approve", "edit", "reject"]},
                "save_shortlist": True,
                "search_cvs": False,                                 # reads: no approval
            }),
        ],
    )


def ask_human(request: dict, auto: bool) -> list[dict]:
    """One decision per pending action. In a product this is a button in Slack or your admin UI."""
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
    """Run until the agent finishes; every interrupt goes to a human, then we resume."""
    result = await agent.ainvoke(payload, cfg)
    while result.get("__interrupt__"):
        request = result["__interrupt__"][0].value                   # {"action_requests": [...], ...}
        result = await agent.ainvoke(Command(resume={"decisions": ask_human(request, auto)}), cfg)
    return result


async def run(question, auto=False, sqlite=None, resume=None):
    thread = resume or f"hire-{uuid.uuid4().hex[:6]}"
    cfg = {"configurable": {"thread_id": thread}}
    print(f"thread_id = {thread}   (Ctrl+C at a prompt, then --sqlite FILE --resume {thread})")
    async with rag_tools() as tools:
        if sqlite:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
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
    if resume:                                                       # continue a pause from an earlier run
        state = await agent.aget_state(cfg)
        if not state.interrupts:
            sys.exit("Nothing to resume on that thread.")
        payload = Command(resume={"decisions": ask_human(state.interrupts[0].value, auto)})
    else:
        payload = {"messages": [{"role": "user", "content": question}]}
    return await drive(agent, payload, cfg, auto)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="*")
    p.add_argument("--auto", action="store_true", help="approve every action (for tests)")
    p.add_argument("--sqlite", help="a SQLite file for durable checkpoints, e.g. checkpoints.db")
    p.add_argument("--resume", help="a thread id to resume (with --sqlite)")
    a = p.parse_args()
    q = " ".join(a.question) or "Find a Kafka candidate and book an interview for the backend job."
    asyncio.run(run(q, a.auto, a.sqlite, a.resume))
