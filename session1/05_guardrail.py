"""
SESSION 1 · STEP 5 · Your own middleware: rules in CODE, not in the prompt.

    python session1/05_guardrail.py "Book an interview for cand-006.md for the backend job"

Two custom middlewares:
  evidence_first   a write about a candidate is BLOCKED until the agent has read that CV with
                   get_cv in this conversation. ("No action without evidence.")
  audit_log        every tool call, its arguments and how long it took → outbox/audit.jsonl

A prompt can be talked out of a rule (or ignore it). Middleware cannot: it runs on every call.
"""
import asyncio
import json
import pathlib
import sys
import time

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


@wrap_tool_call
async def evidence_first(request, handler):
    call = request.tool_call
    if call["name"] in WRITES:
        cv = call["args"].get("cv", "")
        read = any(isinstance(m, ToolMessage) and m.name == "get_cv" and cv and cv in str(m.content)
                   for m in request.state["messages"])
        if not read:
            return ToolMessage(f"Blocked: read the CV first with get_cv('{cv}'), then explain why this "
                               f"candidate fits before you {call['name']}.", tool_call_id=call["id"], name=call["name"])
    return await handler(request)                      # run the real tool


@wrap_tool_call
async def audit_log(request, handler):
    start = time.perf_counter()
    result = await handler(request)
    config.OUTBOX.mkdir(exist_ok=True)
    with open(config.OUTBOX / "audit.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"tool": request.tool_call["name"], "args": request.tool_call["args"],
                            "ms": round((time.perf_counter() - start) * 1000)}, ensure_ascii=False) + "\n")
    return result


async def run(question: str):
    async with rag_tools() as tools:
        agent = create_agent(get_model(), tools + actions.ALL, system_prompt=HIRING_AGENT,
                             middleware=[audit_log, evidence_first])   # order: outer → inner
        result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    banner("Guardrails: evidence_first + audit_log")
    trace(result["messages"])
    print(f"\nAudit log: {config.OUTBOX / 'audit.jsonl'}")
    return result


if __name__ == "__main__":
    asyncio.run(run(" ".join(sys.argv[1:]) or "Book an interview for a Kafka candidate for the backend job."))
