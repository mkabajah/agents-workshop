"""SESSION 1 · STEP 5 · Your own middleware: rules in CODE, not in the prompt.

What you learn
    How to write middleware yourself. A function decorated with @wrap_tool_call runs AROUND every
    tool call: it can look at the call, block it, change it, time it, or log it.

How to run
    python session1/05_guardrail.py "Book an interview for cand-006.md for the backend job"
    MODEL=fake python session1/05_guardrail.py                            # offline, no API key

Two custom middlewares
  evidence_first   a write about a candidate is BLOCKED until the agent has read that CV with
                   get_cv in this conversation. ("No action without evidence.")
  audit_log        every tool call, its arguments and how long it took -> outbox/audit.jsonl

What you'll see
    In the trace, a tool result starting with "Blocked: read the CV first..." when the agent tried to
    book too early; usually the model then reads the CV and tries again. Then the audit log path.

Why code and not the prompt? A prompt can be talked out of a rule (or ignore it). Middleware cannot:
it runs on every call. Comes after step 4 (built-in middleware); step 6 changes approach entirely.

New to async/await, "async with" or decorators? Read docs/python-notes.md (5 minutes).
"""
import asyncio
import json
import pathlib
import sys
import time

# Let Python find the shared "workshop" package in the repo root (docs/python-notes.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

from workshop import actions, config
from workshop.mcp_tools import rag_tools
from workshop.models import get_model
from workshop.prompts import HIRING_AGENT
from workshop.show import banner, trace

WRITES = {"schedule_interview", "send_email"}


# @wrap_tool_call turns this function into middleware that runs AROUND every tool call.
# request = the call the model asked for (+ the conversation so far); handler = "run the real tool".
# It is async because handler() is: we await it to run the tool.
@wrap_tool_call
async def evidence_first(request, handler):
    """Block a booking or an email unless the agent has already read that CV with get_cv."""
    call = request.tool_call
    if call["name"] in WRITES:
        cv = call["args"].get("cv", "")
        # Look back through the conversation for a get_cv result that mentions this CV file.
        read = any(isinstance(m, ToolMessage) and m.name == "get_cv" and cv and cv in str(m.content)
                   for m in request.state["messages"])
        if not read:
            # Returning a message INSTEAD of calling handler() blocks the tool. The model reads why.
            return ToolMessage(f"Blocked: read the CV first with get_cv('{cv}'), then explain why this "
                               f"candidate fits before you {call['name']}.", tool_call_id=call["id"], name=call["name"])
    return await handler(request)                      # run the real tool


@wrap_tool_call
async def audit_log(request, handler):
    """Run the tool, then write one JSON line per call to outbox/audit.jsonl (tool, arguments, time)."""
    start = time.perf_counter()
    result = await handler(request)
    config.OUTBOX.mkdir(exist_ok=True)
    with open(config.OUTBOX / "audit.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"tool": request.tool_call["name"], "args": request.tool_call["args"],
                            "ms": round((time.perf_counter() - start) * 1000)}, ensure_ascii=False) + "\n")
    return result


async def run(question: str):
    """Run the agent with both guardrails for one request and print the trace."""
    # Start the CV MCP server once. "async with" stops it again at the end, even after an error.
    async with rag_tools() as tools:
        agent = create_agent(get_model(), tools + actions.ALL, system_prompt=HIRING_AGENT,
                             middleware=[audit_log, evidence_first])   # order: outer → inner
        # audit_log is the outer layer, so it also logs the calls that evidence_first blocks.
        result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    banner("Guardrails: evidence_first + audit_log")
    trace(result["messages"])
    print(f"\nAudit log: {config.OUTBOX / 'audit.jsonl'}")
    return result


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    # asyncio.run() starts the async function and waits until it is finished.
    asyncio.run(run(" ".join(sys.argv[1:]) or "Book an interview for a Kafka candidate for the backend job."))
