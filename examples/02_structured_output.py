"""
EXAMPLE 02 · Structured output: the agent returns a typed object, not prose.   Pattern: agent → data

    python examples/02_structured_output.py "Summarise cand-006.md for the backend job"

What it shows: response_format=<Pydantic model> makes create_agent finish with an OBJECT (fields you
defined) instead of free text. The agent still uses its tools first; only the final answer is typed.

    question ──▶ agent loop (tools…) ──▶ CandidateSummary(cv=..., must_haves_met=[...], ...)

When you'd use it: whenever the next step is code, not a person: save to a database, sort a table,
fill a UI card, trigger an email only if recommend_interview is True.
What you'll see: the summary printed as JSON.
Needs a real model (MODEL=fake cannot fill a schema).

New to async/await, "async with" or decorators? Read docs/python-notes.md (5 minutes).
"""
import asyncio
import pathlib
import sys

# Let Python find the shared "workshop" package in the repo root (docs/python-notes.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from pydantic import BaseModel, Field

from workshop import config
from workshop.mcp_tools import rag_tools
from workshop.models import get_model
from workshop.prompts import HIRING_AGENT


class CandidateSummary(BaseModel):
    """The shape of the answer we want back. Field descriptions are sent to the model as instructions."""

    cv: str = Field(description="CV file name")
    must_haves_met: list[str] = Field(description="Must-haves with evidence in the CV")
    missing: list[str] = Field(description="Must-haves with no evidence")
    risk: str = Field(description="One job-relevant risk")
    recommend_interview: bool


# async def: this function waits on the model and the MCP server; each wait is marked with await.
async def run(question: str):
    """Run the agent and return its final answer as a CandidateSummary object."""
    if config.MODEL == "fake":
        print("Structured output needs a real model: set MODEL in .env.")
        return None
    # Start the CV MCP server once. "async with" stops it again at the end, even after an error.
    async with rag_tools() as tools:
        # response_format = "end with this schema": the model fills it and the result is validated.
        agent = create_agent(get_model(), tools, system_prompt=HIRING_AGENT, response_format=CandidateSummary)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    # The typed object lives next to the messages, under "structured_response".
    summary: CandidateSummary = result["structured_response"]
    print(summary.model_dump_json(indent=2))
    return summary


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    # asyncio.run() starts the async function and waits until it is finished.
    asyncio.run(run(" ".join(sys.argv[1:]) or "Summarise cand-006.md for the backend job"))
