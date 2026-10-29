"""SESSION 2 · STEP 10 · Tracing: see every model call and tool call, with time and tokens.

What you learn
    How to look inside a running agent. LangChain calls "callbacks" at every event (a model call
    starts or ends, a tool starts or ends). A tracer is a callback handler that records them. Here we
    build a tiny one that prints a table; LangSmith does the same thing with a web page.

How to run
    python session2/10_tracing.py                    # a local trace table, no account needed
    MODEL=fake python session2/10_tracing.py         # offline, no API key
    # with LangSmith (free tier): put these in .env, then run the same command
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=lsv2_...
    LANGSMITH_PROJECT=agents-workshop

What you'll see
    One row per step, indented by depth: the lead's model and tool calls, and INSIDE them the
    specialists' own calls, each with seconds and input/output tokens. Slow or expensive steps stand out.

It runs the multi-agent team from step 7 (approvals are auto-approved for the demo).
Strands and CrewAI emit OpenTelemetry traces instead; LangSmith, Langfuse and others accept them.

New to async/await, "async with" or decorators? Read docs/python-notes.md (5 minutes).
"""
import asyncio
import importlib.util
import os
import pathlib
import sys
import time
import uuid

# Let Python find the shared "workshop" package in the repo root (docs/python-notes.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.callbacks import AsyncCallbackHandler
from langgraph.types import Command

from workshop.show import banner

# Load step 7 as a module so we can reuse its build(). (A file name starting with a digit cannot be
# imported with a normal `import` line.)
spec = importlib.util.spec_from_file_location("supervisor", pathlib.Path(__file__).with_name("07_supervisor.py"))
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


class TraceTable(AsyncCallbackHandler):
    """A tiny tracer: what LangSmith shows, as text.

    LangChain calls these on_* methods at each event. Every event has a run_id and the run_id of its
    parent, so we can rebuild the tree: lead -> specialist tool -> specialist's model call."""

    def __init__(self):
        # rows: finished steps · start: when each run began · parent: run -> parent run · names: what it is
        self.rows, self.start, self.parent, self.names = [], {}, {}, {}

    def _depth(self, run_id):
        """How deep a step is: count the model/tool ancestors above it (chains in between don't count)."""
        d, p = 0, self.parent.get(run_id)
        while p is not None:
            d, p = d + (p in self.names), self.parent.get(p)
        return d

    async def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kw):
        """A graph step started: remember its parent so the tree stays connected."""
        self.parent[run_id] = parent_run_id

    async def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kw):
        """A model call started: note the time."""
        self.parent[run_id], self.start[run_id], self.names[run_id] = parent_run_id, time.perf_counter(), "model"

    async def on_llm_end(self, response, *, run_id, **kw):
        """A model call ended: record what it decided (answer or tool request), time and tokens."""
        msg = response.generations[0][0].message
        usage = getattr(msg, "usage_metadata", None) or {}
        calls = [tc["name"] for tc in getattr(msg, "tool_calls", [])]
        what = f"→ asks {', '.join(calls)}" if calls else "→ answers"
        self.rows.append((self._depth(run_id), "🤖 model", what, time.perf_counter() - self.start[run_id],
                          usage.get("input_tokens", "-"), usage.get("output_tokens", "-")))

    async def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kw):
        """A tool call started: note the time and the tool name."""
        self.parent[run_id], self.start[run_id], self.names[run_id] = parent_run_id, time.perf_counter(), serialized.get("name")

    async def on_tool_end(self, output, *, run_id, **kw):
        """A tool call ended: record it with its duration."""
        self.rows.append((self._depth(run_id), "🔧 tool", self.names[run_id], time.perf_counter() - self.start[run_id], "", ""))

    def print(self):
        """Print the table, one row per step, indented by depth."""
        print(f"{'step':<46}{'seconds':>8}{'in tok':>9}{'out tok':>9}")
        for depth, kind, what, secs, tin, tout in self.rows:         # in the order steps finished
            label = "   " * depth + f"{kind} {what}"
            print(f"{label[:45]:<46}{secs:>8.2f}{str(tin):>9}{str(tout):>9}")


async def run(question="Shortlist 3 candidates for the backend job and book an interview with the best one."):
    """Run the step-7 team once with our tracer attached, then print the trace table."""
    tracer = TraceTable()
    # "callbacks" in the config reach every nested call, so the specialists are traced too.
    # run_name, tags and metadata are labels you can filter on in LangSmith.
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


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    # asyncio.run() starts the async function and waits until it is finished.
    asyncio.run(run())
