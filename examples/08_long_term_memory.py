"""
EXAMPLE 08 · Long-term memory across conversations.       Pattern: a store + remember/recall tools

    python examples/08_long_term_memory.py

Thread memory (checkpointer) ends with the conversation. A STORE keeps facts across threads,
per user: "Dana prefers candidates in Haifa". The agent reads and writes it through two tools,
so it decides what is worth remembering. Production: a Postgres or vector store instead of memory.
"""
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langgraph.store.memory import InMemoryStore

from workshop.mcp_tools import rag_tools
from workshop.models import get_model
from workshop.prompts import HIRING_AGENT
from workshop.show import banner, trace

store = InMemoryStore()


@tool
def remember_preference(note: str, runtime: ToolRuntime) -> str:
    """Save a lasting preference of this recruiter (e.g. 'prefers candidates in Haifa').
    Use when the recruiter states a preference that should apply to future searches."""
    user = runtime.context["user_id"]
    key = f"pref-{len(store.search((user, 'prefs'))) + 1}"
    store.put((user, "prefs"), key, {"note": note})
    return f"Saved for {user}: {note}"


@tool
def recall_preferences(question: str, runtime: ToolRuntime) -> list[str]:
    """Return this recruiter's saved preferences. Call it at the start of a new search."""
    return [item.value["note"] for item in store.search((runtime.context["user_id"], "prefs"))]


async def run():
    async with rag_tools() as tools:
        agent = create_agent(get_model(), tools + [remember_preference, recall_preferences],
                             system_prompt=HIRING_AGENT + "\nRecall the recruiter's preferences before searching.",
                             store=store, context_schema=dict)
        for thread, text in [("monday", "Remember my preference: I prefer candidates in Haifa."),
                             ("tuesday", "Recall my preferences, then who knows Kubernetes?")]:
            out = await agent.ainvoke({"messages": [{"role": "user", "content": text}]},
                                      {"configurable": {"thread_id": thread}}, context={"user_id": "dana"})
            banner(f"thread {thread} (a new conversation)")
            trace(out["messages"])
    print("\nStore now holds:", [i.value for i in store.search(("dana", "prefs"))])
    return store


if __name__ == "__main__":
    asyncio.run(run())
