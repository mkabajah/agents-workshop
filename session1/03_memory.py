"""SESSION 1 · STEP 3 · SHORT-TERM memory: a checkpointer saves the conversation after every step, per thread.

Short-term or long-term memory?
    SHORT-TERM (this step)  = the conversation itself: the messages of ONE thread. It lets the agent
                              understand "them" in a follow-up. A new thread starts empty.
                              LangChain: a checkpointer + thread_id.
    LONG-TERM (examples/08) = facts that outlive a conversation, e.g. "Dana prefers Haifa".
                              Saved in a store, read and written through tools, shared by all of
                              Dana's threads. LangChain: a store (InMemoryStore, Postgres…).

What you learn
    How an agent remembers. A checkpointer stores the message list of each conversation ("thread")
    after every step. The thread_id you pass decides which conversation you continue.

How to run
    python session1/03_memory.py
    MODEL=fake python session1/03_memory.py                               # offline, no API key

What happens
    Turn 1 (thread A):  "Who knows Kafka?"
    Turn 2 (thread A):  "Which of them lives in Haifa?"      <- needs turn 1: "them" = the Kafka people
    Turn 3 (thread B):  "Which of them lives in Haifa?"      <- a new thread has no memory of turn 1

What you'll see
    One trace per turn and how many messages each thread has stored. Thread A keeps growing;
    thread B starts from zero.

Comes after step 2: same agent, one extra argument (checkpointer). Step 4 uses the same
checkpointer to pause the agent for human approval.
Production: swap InMemorySaver for SqliteSaver / PostgresSaver so a conversation survives a restart.

New to async/await, "async with" or decorators? Read docs/python-notes.md (5 minutes).
"""
import asyncio
import pathlib
import sys

# Let Python find the shared "workshop" package in the repo root (docs/python-notes.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from workshop.mcp_tools import rag_tools
from workshop.models import get_model
from workshop.prompts import HIRING_AGENT
from workshop.show import banner, trace


async def run():
    """Ask three questions on two threads and show that only the same thread remembers."""
    # Start the CV MCP server once. "async with" stops it again at the end, even after an error.
    async with rag_tools() as tools:
        # checkpointer = SHORT-TERM memory: after every step it saves the messages of each thread.
        # InMemorySaver keeps them in RAM (gone when the script ends); SQLite/Postgres keep them on disk.
        agent = create_agent(get_model(), tools, system_prompt=HIRING_AGENT, checkpointer=InMemorySaver())

        # A thread_id is like a chat window: same id = same conversation, new id = fresh start.
        thread_a = {"configurable": {"thread_id": "recruiter-dana"}}
        thread_b = {"configurable": {"thread_id": "recruiter-omar"}}

        for cfg, text in [(thread_a, "Who knows Kafka?"),
                          (thread_a, "Which of them lives in Haifa?"),
                          (thread_b, "Which of them lives in Haifa?")]:
            # aget_state reads what the checkpointer has saved for this thread so far.
            # We note the count so the trace below prints only the NEW messages of this turn.
            before = len((await agent.aget_state(cfg)).values.get("messages", []))
            # We send ONLY the new question; the checkpointer adds the earlier messages of this thread.
            result = await agent.ainvoke({"messages": [{"role": "user", "content": text}]}, cfg)
            banner(f"thread {cfg['configurable']['thread_id']} · {len(result['messages'])} messages stored")
            trace(result["messages"], start=before)

        state = await agent.aget_state(thread_a)                    # read back what the checkpointer saved for Dana
        print(f"\nThread A has {len(state.values['messages'])} messages saved. Thread B starts from zero.")
        print("This is SHORT-TERM memory: it lives inside one thread. For facts that last across threads\n"
              "(long-term memory), see examples/08_long_term_memory.py.")
        return state


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    # asyncio.run() starts the async function and waits until it is finished.
    asyncio.run(run())
