"""
SESSION 1 · STEP 3 · Memory: a checkpointer saves the conversation after every step, per thread.

    python session1/03_memory.py

Turn 1 (thread A):  "Who knows Kafka?"
Turn 2 (thread A):  "Which of them lives in Haifa?"      ← needs turn 1: "them" = the Kafka people
Turn 3 (thread B):  "Which of them lives in Haifa?"      ← a new thread has no memory of turn 1

Production: swap InMemorySaver for SqliteSaver / PostgresSaver so a conversation survives a restart.

New to async/await, "async with" or decorators? Read PYTHON_NOTES.md (5 minutes).
"""
import asyncio
import pathlib
import sys

# Let Python find the shared "workshop" package in the repo root (PYTHON_NOTES.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from workshop.mcp_tools import rag_tools
from workshop.models import get_model
from workshop.prompts import HIRING_AGENT
from workshop.show import banner, trace


# async def: this function waits on the model and the MCP server; each wait is marked with await.
async def run():
    # Start the CV MCP server once. "async with" stops it again at the end, even after an error.
    async with rag_tools() as tools:
        # checkpointer = memory: after every step it saves the messages of each thread.
        # InMemorySaver keeps them in RAM (gone when the script ends); SQLite/Postgres keep them on disk.
        agent = create_agent(get_model(), tools, system_prompt=HIRING_AGENT, checkpointer=InMemorySaver())

        # A thread_id is like a chat window: same id = same conversation, new id = fresh start.
        thread_a = {"configurable": {"thread_id": "recruiter-dana"}}
        thread_b = {"configurable": {"thread_id": "recruiter-omar"}}

        for cfg, text in [(thread_a, "Who knows Kafka?"),
                          (thread_a, "Which of them lives in Haifa?"),
                          (thread_b, "Which of them lives in Haifa?")]:
            before = len((await agent.aget_state(cfg)).values.get("messages", []))
            result = await agent.ainvoke({"messages": [{"role": "user", "content": text}]}, cfg)
            banner(f"thread {cfg['configurable']['thread_id']} · {len(result['messages'])} messages stored")
            trace(result["messages"], start=before)

        state = await agent.aget_state(thread_a)                    # read back what the checkpointer saved for Dana
        print(f"\nThread A has {len(state.values['messages'])} messages saved. Thread B starts from zero.")
        return state


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    # asyncio.run() starts the async function and waits until it is finished.
    asyncio.run(run())
