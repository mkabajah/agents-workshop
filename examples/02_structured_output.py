"""
EXAMPLE 02 · Structured output: the agent returns a typed object, not prose.   Pattern: agent → data

    python examples/02_structured_output.py "Summarise cand-006.md for the backend job"

response_format=<Pydantic model> makes create_agent finish with an object your code can store,
sort or show in a UI. Use it whenever the next step is code, not a person.
Needs a real model (MODEL=fake cannot fill a schema).
"""
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from pydantic import BaseModel, Field

from workshop import config
from workshop.mcp_tools import rag_tools
from workshop.models import get_model
from workshop.prompts import HIRING_AGENT


class CandidateSummary(BaseModel):
    cv: str = Field(description="CV file name")
    must_haves_met: list[str] = Field(description="Must-haves with evidence in the CV")
    missing: list[str] = Field(description="Must-haves with no evidence")
    risk: str = Field(description="One job-relevant risk")
    recommend_interview: bool


async def run(question: str):
    if config.MODEL == "fake":
        print("Structured output needs a real model: set MODEL in .env.")
        return None
    async with rag_tools() as tools:
        agent = create_agent(get_model(), tools, system_prompt=HIRING_AGENT, response_format=CandidateSummary)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    summary: CandidateSummary = result["structured_response"]
    print(summary.model_dump_json(indent=2))
    return summary


if __name__ == "__main__":
    asyncio.run(run(" ".join(sys.argv[1:]) or "Summarise cand-006.md for the backend job"))
