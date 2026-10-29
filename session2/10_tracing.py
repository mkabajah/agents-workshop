"""
SESSION 2 · STEP 10 · Tracing: see every model call and tool call, with time and tokens.

    python session2/10_tracing.py                    # a local trace table, no account needed
    # with LangSmith (free tier): put these in .env, then run the same command
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=lsv2_...
    LANGSMITH_PROJECT=agents-workshop

It runs the multi-agent team from step 7 and prints one row per step, indented by depth:
the lead's calls, and INSIDE them the specialists' own model and tool calls.
Strands and CrewAI emit OpenTelemetry traces instead; LangSmith, Langfuse and others accept them.
"""
import asyncio
import importlib.util
import os
import pathlib
import sys
import time
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.callbacks import AsyncCallbackHandler
from langgraph.types import Command

from workshop.show import banner

spec = importlib.util.spec_from_file_location("supervisor", pathlib.Path(__file__).with_name("07_supervisor.py"))
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


class TraceTable(AsyncCallbackHandler):
    """A tiny tracer: what LangSmith shows, as text."""

    def __init__(self):
        self.rows, self.start, self.parent, self.names = [], {}, {}, {}

    def _depth(self, run_id):
        d, p = 0, self.parent.get(run_id)
        while p is not None:
            d, p = d + (p in self.names), self.parent.get(p)
        return d

    async def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kw):
        self.parent[run_id] = parent_run_id

    async def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kw):
        self.parent[run_id], self.start[run_id], self.names[run_id] = parent_run_id, time.perf_counter(), "model"

    async def on_llm_end(self, response, *, run_id, **kw):
        msg = response.generations[0][0].message
        usage = getattr(msg, "usage_metadata", None) or {}
        calls = [tc["name"] for tc in getattr(msg, "tool_calls", [])]
        what = f"→ asks {', '.join(calls)}" if calls else "→ answers"
        self.rows.append((self._depth(run_id), "🤖 model", what, time.perf_counter() - self.start[run_id],
                          usage.get("input_tokens", "-"), usage.get("output_tokens", "-")))

    async def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kw):
        self.parent[run_id], self.start[run_id], self.names[run_id] = parent_run_id, time.perf_counter(), serialized.get("name")

    async def on_tool_end(self, output, *, run_id, **kw):
        self.rows.append((self._depth(run_id), "🔧 tool", self.names[run_id], time.perf_counter() - self.start[run_id], "", ""))

    def print(self):
        print(f"{'step':<46}{'seconds':>8}{'in tok':>9}{'out tok':>9}")
        for depth, kind, what, secs, tin, tout in self.rows:         # in the order steps finished
            label = "   " * depth + f"{kind} {what}"
            print(f"{label[:45]:<46}{secs:>8.2f}{str(tin):>9}{str(tout):>9}")


async def run(question="Shortlist 3 candidates for the backend job and book an interview with the best one."):
    tracer = TraceTable()
    cfg = {"configurable": {"thread_id": f"traced-{uuid.uuid4().hex[:6]}"}, "callbacks": [tracer],
           "run_name": "hiring-team", "tags": ["session2", "supervisor"], "metadata": {"job": "backend-engineer"}}
    async with supervisor.rag_tools() as tools:
        lead = supervisor.build(tools)
        result = await lead.ainvoke({"messages": [{"role": "user", "content": question}]}, cfg)
        while result.get("__interrupt__"):                            # auto-approve for the demo
            n = len(result["__interrupt__"][0].value["action_requests"])
            result = await lead.ainvoke(Command(resume={"decisions": [{"type": "approve"}] * n}), cfg)
    banner("Trace (indented = inside a specialist)")
    tracer.print()
    if os.getenv("LANGSMITH_TRACING") == "true":
        print(f"\nLangSmith: open project '{os.getenv('LANGSMITH_PROJECT', 'default')}' at smith.langchain.com")
    return tracer


if __name__ == "__main__":
    asyncio.run(run())
