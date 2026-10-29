"""
EXAMPLE 01 · Tool calling, one call, no loop.            Pattern: the atom every agent is built from

    python examples/01_tool_calling_basics.py "Which candidates know Kafka?"

The model gets a question and the tool schemas, and returns either text or a request to call a
tool. Nothing runs: you only SEE the request. Everything else in this repo is this, in a loop.
"""
import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from workshop.mcp_tools import rag_tools
from workshop.models import get_model


async def run(question: str):
    async with rag_tools() as tools:
        model = get_model().bind_tools(tools)              # the model now knows the 4 tool schemas
        reply = await model.ainvoke(question)
    if reply.tool_calls:
        for call in reply.tool_calls:
            print("The model ASKS to call:", call["name"], json.dumps(call["args"]))
        print("\nNothing has run yet. Your code decides whether to run it (that is the agent loop).")
    else:
        print("The model answered directly:", reply.content)
    return reply


if __name__ == "__main__":
    asyncio.run(run(" ".join(sys.argv[1:]) or "Which candidates know Kafka?"))
