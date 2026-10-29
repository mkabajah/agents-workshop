"""
EXAMPLE 12 · The SAME AgentCore entrypoint around a LangChain/LangGraph agent.      AWS, any framework

    pip install bedrock-agentcore
    python examples/12_agentcore_langgraph.py
    curl -X POST localhost:8080/invocations -H 'Content-Type: application/json' \\
         -d '{"prompt": "Who knows Kafka?", "thread_id": "dana-1"}'

The point: AgentCore Runtime hosts LangGraph, Strands, CrewAI or plain Python the same way.
Choosing LangGraph today does not block running on AWS tomorrow; only these ~15 lines change.

New to async/await, "async with" or decorators? Read PYTHON_NOTES.md (5 minutes).
"""
import pathlib
import sys

# Let Python find the shared "workshop" package in the repo root (PYTHON_NOTES.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from workshop.mcp_tools import rag_tools
from workshop.models import get_model
from workshop.prompts import HIRING_AGENT
from workshop.show import content_text

app = BedrockAgentCoreApp()
memory = InMemorySaver()               # on AWS: AgentCore Memory or a Postgres checkpointer


@app.entrypoint
async def invoke(payload: dict) -> dict:
    prompt = str(payload.get("prompt", ""))[:2000]
    if not prompt:
        return {"error": "Send {'prompt': '...'}"}
    cfg = {"configurable": {"thread_id": str(payload.get("thread_id", "default"))}}
    # Start the CV MCP server once. "async with" stops it again at the end, even after an error.
    async with rag_tools() as tools:
        agent = create_agent(get_model(), tools, system_prompt=HIRING_AGENT, checkpointer=memory)
        out = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]}, cfg)
    return {"answer": content_text(out["messages"][-1]),
            "tools_used": [c["name"] for m in out["messages"] for c in getattr(m, "tool_calls", [])]}


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    app.run()
