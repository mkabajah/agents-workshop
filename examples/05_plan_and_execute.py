"""
EXAMPLE 05 · Plan, then execute.        Pattern: planner + executor (good for long multi-step tasks)

    python examples/05_plan_and_execute.py "Compare the Kafka candidates for the backend job"

1. A planner writes a short numbered plan (no tools).
2. An executor agent runs each step with the tools, carrying the earlier results forward.
3. A final call writes the answer from the step results.
Why: a visible plan is easy to review (even to approve) and keeps long tasks on track.
With MODEL=fake the plan is a fixed template.
"""
import asyncio
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent

from workshop import config
from workshop.mcp_tools import rag_tools
from workshop.models import get_model
from workshop.prompts import HIRING_AGENT
from workshop.show import banner, content_text, short


async def make_plan(task: str) -> list[str]:
    if config.MODEL == "fake":
        return ["Read the backend job's must-haves with list_jobs",
                "Find every candidate who knows Kafka",
                "Check the evidence in the CV of the first Kafka candidate"]
    reply = await get_model().ainvoke(
        "Write a plan of at most 4 numbered steps for this recruiting task. One tool-sized action per "
        f"step (list_jobs, find_candidates, search_cvs, get_cv). No prose.\n\nTask: {task}")
    return [re.sub(r"^\s*\d+[.)]\s*", "", line).strip() for line in content_text(reply).splitlines() if line.strip()][:4]


async def run(task: str):
    plan = await make_plan(task)
    banner("Plan")
    for i, step in enumerate(plan, 1):
        print(f"{i}. {step}")
    notes = []
    async with rag_tools() as tools:
        executor = create_agent(get_model(), tools, system_prompt=HIRING_AGENT)
        for i, step in enumerate(plan, 1):
            context = "\n".join(notes[-3:])
            out = await executor.ainvoke({"messages": [{"role": "user",
                                          "content": f"Earlier results:\n{context}\n\nDo step {i}: {step}"}]})
            notes.append(f"Step {i} ({step}): {content_text(out['messages'][-1])}")
            print(f"\n✓ step {i}:", short(notes[-1], 240))
    final = await get_model().ainvoke(f"Task: {task}\n\nStep results:\n" + "\n".join(notes) +
                                      "\n\nWrite the final answer, citing CV file names.")
    banner("Answer")
    print(content_text(final))
    return plan, notes


if __name__ == "__main__":
    asyncio.run(run(" ".join(sys.argv[1:]) or "Compare the Kafka candidates for the backend job"))
