"""
EXAMPLE 08 · Long-term memory across conversations.       Pattern: a store + remember/recall tools

    python examples/08_long_term_memory.py

What it shows: thread memory (a checkpointer, session1/03) ends with the conversation. A STORE keeps facts
across threads, per user: "Dana prefers candidates in Haifa". The agent reads and writes it through two
tools, so the model decides what is worth remembering.

    Monday thread:  "I prefer Haifa"      ──▶ remember_preference ──▶ store[("dana", "prefs")]
    Tuesday thread: "who knows Kubernetes?" ─▶ recall_preferences  ◀── store[("dana", "prefs")]

When you'd use it: user preferences, past decisions, facts a team taught the agent.
What you'll see: two separate conversations, and what the store holds at the end.
Production: a Postgres or vector-backed store instead of InMemoryStore.

New to async/await, "async with" or decorators? Read docs/python-notes.md (5 minutes).
"""
import asyncio
import pathlib
import sys

# Let Python find the shared "workshop" package in the repo root (docs/python-notes.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langgraph.store.memory import InMemoryStore

from workshop.mcp_tools import rag_tools
from workshop.models import get_model
from workshop.prompts import HIRING_AGENT
from workshop.show import banner, trace

# The store is a key-value database organised by "namespace" tuples, e.g. ("dana", "prefs").
# InMemoryStore lives only while this script runs; a real app uses a persistent store.
store = InMemoryStore()


@tool
def remember_preference(note: str, runtime: ToolRuntime) -> str:
    """Save a lasting preference of this recruiter (e.g. 'prefers candidates in Haifa').
    Use when the recruiter states a preference that should apply to future searches."""
    # runtime is injected by LangChain (the model never sees it): it carries the per-call context.
    user = runtime.context["user_id"]
    key = f"pref-{len(store.search((user, 'prefs'))) + 1}"
    store.put((user, "prefs"), key, {"note": note})
    return f"Saved for {user}: {note}"


@tool
def recall_preferences(question: str, runtime: ToolRuntime) -> list[str]:
    """Return this recruiter's saved preferences. Call it at the start of a new search."""
    return [item.value["note"] for item in store.search((runtime.context["user_id"], "prefs"))]


# async def: this function waits on the model and the MCP server; each wait is marked with await.
async def run():
    """Two conversations (Monday, Tuesday) for the same user: the second one recalls the first."""
    # Start the CV MCP server once. "async with" stops it again at the end, even after an error.
    async with rag_tools() as tools:
        agent = create_agent(get_model(), tools + [remember_preference, recall_preferences],
                             system_prompt=HIRING_AGENT + "\nRecall the recruiter's preferences before searching.",
                             store=store, context_schema=dict)   # context = per-call data, like who is asking
        for thread, text in [("monday", "Remember my preference: I prefer candidates in Haifa."),
                             ("tuesday", "Recall my preferences, then who knows Kubernetes?")]:
            # A different thread_id each day = a fresh conversation; context tells the tools who the user is.
            out = await agent.ainvoke({"messages": [{"role": "user", "content": text}]},
                                      {"configurable": {"thread_id": thread}}, context={"user_id": "dana"})
            banner(f"thread {thread} (a new conversation)")
            trace(out["messages"])
    print("\nStore now holds:", [i.value for i in store.search(("dana", "prefs"))])
    return store


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    # asyncio.run() starts the async function and waits until it is finished.
    asyncio.run(run())
