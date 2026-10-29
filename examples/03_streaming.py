"""
EXAMPLE 03 · Streaming: show each step as it happens.      Pattern: UX for slow agents

    python examples/03_streaming.py "Who knows Kubernetes and lives in Haifa?"

An agent can take 10–60 seconds. Streaming the steps ("searching CVs… reading cand-004…")
makes the wait feel short and makes the agent's reasoning visible to the user.
stream_mode="updates" yields one event per node: the model's turn, then the tools' turn.
"""
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent

from workshop.mcp_tools import rag_tools
from workshop.models import get_model
from workshop.prompts import HIRING_AGENT
from workshop.show import content_text, short


async def run(question: str):
    steps = 0
    async with rag_tools() as tools:
        agent = create_agent(get_model(), tools, system_prompt=HIRING_AGENT)
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


if __name__ == "__main__":
    asyncio.run(run(" ".join(sys.argv[1:]) or "Who knows Kubernetes and lives in Haifa?"))
