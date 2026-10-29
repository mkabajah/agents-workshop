"""
EXAMPLE 01 · Tool calling, one call, no loop.            Pattern: the atom every agent is built from

    python examples/01_tool_calling_basics.py "Which candidates know Kafka?"

What it shows: the model gets a question plus the tool descriptions ("schemas") and returns EITHER
text OR a request to call a tool. Nothing runs: you only SEE the request. Every agent in this repo
is this one call, repeated in a loop by your code.

    question + tool schemas ──▶ model ──▶ "please call find_candidates(skill='kafka')"   (not run!)

When you'd use it: to debug tool descriptions ("why does the model pick the wrong tool?").
What you'll see: "The model ASKS to call: find_candidates {...}".

New to async/await, "async with" or decorators? Read docs/python-notes.md (5 minutes).
"""
import asyncio
import json
import pathlib
import sys

# Let Python find the shared "workshop" package in the repo root (docs/python-notes.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from workshop.mcp_tools import rag_tools
from workshop.models import get_model


# async def: this function waits on the model and the MCP server; each wait is marked with await.
async def run(question: str):
    """Ask the model once with the CV tools attached, and print what it wants to do."""
    # Start the CV MCP server once. "async with" stops it again at the end, even after an error.
    async with rag_tools() as tools:
        # bind_tools() only TELLS the model which tools exist (name, description, arguments).
        # It does not give the model the power to run them: that stays with your code.
        model = get_model().bind_tools(tools)              # the model now knows the 4 tool schemas
        reply = await model.ainvoke(question)
    # reply.tool_calls is a list of requests: [{"name": ..., "args": {...}, "id": ...}]. Empty = plain answer.
    if reply.tool_calls:
        for call in reply.tool_calls:
            print("The model ASKS to call:", call["name"], json.dumps(call["args"]))
        print("\nNothing has run yet. Your code decides whether to run it (that is the agent loop).")
    else:
        print("The model answered directly:", reply.content)
    return reply


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    # asyncio.run() starts the async function and waits until it is finished.
    asyncio.run(run(" ".join(sys.argv[1:]) or "Which candidates know Kafka?"))
