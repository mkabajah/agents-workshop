"""
EXAMPLE 03 · Streaming: show each step as it happens.      Pattern: UX for slow agents

    python examples/03_streaming.py "Who knows Kubernetes and lives in Haifa?"

What it shows: instead of waiting for the whole run (ainvoke), astream() gives you each step the moment
it finishes. An agent can take 10–60 seconds; streaming the steps ("searching CVs… reading cand-004…")
makes the wait feel short and makes the agent's reasoning visible to the user.

    [model] calling find_candidates(...)  →  [tools] result  →  [model] calling get_cv(...)  →  [model] answer

When you'd use it: any chat UI or Slack bot where a person is waiting.
stream_mode="updates" yields one event per node: the model's turn, then the tools' turn.
(Other modes: "messages" streams the answer token by token; "values" gives the full state each step.)

New to async/await, "async with" or decorators? Read docs/python-notes.md (5 minutes).
"""
import asyncio
import pathlib
import sys

# Let Python find the shared "workshop" package in the repo root (docs/python-notes.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent

from workshop.mcp_tools import rag_tools
from workshop.models import get_model
from workshop.prompts import HIRING_AGENT
from workshop.show import content_text, short


# async def: this function waits on the model and the MCP server; each wait is marked with await.
async def run(question: str):
    """Run the agent and print every step as it arrives; return how many messages were streamed."""
    steps = 0
    # Start the CV MCP server once. "async with" stops it again at the end, even after an error.
    async with rag_tools() as tools:
        agent = create_agent(get_model(), tools, system_prompt=HIRING_AGENT)
        # "async for" = loop over results that arrive over time.
        # Each update looks like {node_name: {"messages": [...]}}.
        async for update in agent.astream({"messages": [{"role": "user", "content": question}]},
                                          stream_mode="updates"):
            for node, data in update.items():
                for m in data.get("messages", []):
                    steps += 1
                    if getattr(m, "tool_calls", None):
                        print(f"[{node}] calling", ", ".join(f"{c['name']}({short(c['args'], 80)})" for c in m.tool_calls))
                    else:
                        print(f"[{node}]", short(content_text(m), 200))
    return steps


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    # asyncio.run() starts the async function and waits until it is finished.
    asyncio.run(run(" ".join(sys.argv[1:]) or "Who knows Kubernetes and lives in Haifa?"))
