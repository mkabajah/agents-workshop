"""
SESSION 1 · STEP 2 · The same loop with LangChain 1.x: create_agent + your MCP server.

    python session1/02_first_agent.py "Which candidates know Kafka and Docker? Check the evidence."

create_agent(model, tools, system_prompt) builds the loop from step 1 for you, as a LangGraph
graph with two nodes (model, tools). The tools: your 4 RAG tools over MCP + one local tool.
"""
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent

from workshop import actions
from workshop.mcp_tools import rag_tools
from workshop.models import get_model
from workshop.prompts import HIRING_AGENT
from workshop.show import banner, trace


async def run(question: str):
    async with rag_tools() as tools:                      # search_cvs, get_cv, find_candidates, list_jobs
        agent = create_agent(
            model=get_model(),                            # MODEL in .env: anthropic:…, openai:…, ollama:…
            tools=tools + actions.READ_TOOLS,             # MCP tools and plain @tool functions, one list
            system_prompt=HIRING_AGENT,
        )
        result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})

    calls = [tc["name"] for m in result["messages"] for tc in getattr(m, "tool_calls", [])]
    banner(f"{len(calls)} tool call(s): {calls}")
    trace(result["messages"])
    return result


if __name__ == "__main__":
    asyncio.run(run(" ".join(sys.argv[1:]) or "Which candidates know Kafka and Docker? Check the evidence."))
