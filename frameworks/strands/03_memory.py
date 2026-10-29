"""
STRANDS · STEP 3 · SHORT-TERM memory: one session per conversation.

Short-term or long-term memory?
    SHORT-TERM (this step)  = the conversation itself: the messages of ONE thread. It lets the agent
                              understand "them" in a follow-up. A new thread starts empty.
                              Strands: a session manager + session_id.
    LONG-TERM (examples/08) = facts that outlive a conversation, e.g. "Dana prefers Haifa".
                              Saved in a store, read and written through tools, shared by all of
                              Dana's threads. (Strands: a memory tool or an external store; not shown here.)

What it does
    Two recruiters talk to the agent. Dana asks two questions in the SAME conversation, so the second
    question ("which of THEM lives in Haifa?") makes sense. Omar asks the same follow-up in a NEW
    conversation, so the agent has no idea who "them" is.

How to run
    python run.py 03 --framework strands
    python frameworks/strands/03_memory.py

What you'll see
    Dana's session has more messages than Omar's. The sessions are saved as files under
    outbox/sessions/: run the script again and Dana's conversation continues where it stopped.

Same thing in LangChain: session1/03_memory.py (a checkpointer + one thread_id per conversation).

How memory works in Strands
    An Agent object remembers its own conversation in agent.messages, but only while the program runs.
    A session manager saves those messages somewhere and loads them back next time. Here a
    FileSessionManager writes them to disk. One session_id = one conversation (= a LangGraph thread).
"""
import pathlib
import sys

# Let Python find the shared "workshop" package that lives two folders up (the repo root).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from strands import Agent
from strands.session.file_session_manager import FileSessionManager

from workshop import config
from workshop.prompts import HIRING_AGENT
from workshop.strands_kit import cv_client, strands_model

SESSIONS = config.OUTBOX / "sessions"            # where the saved conversations live


def agent_for(session_id: str, tools) -> Agent:
    """One agent per conversation. The session manager loads earlier messages and saves new ones."""
    return Agent(model=strands_model(), tools=tools, system_prompt=HIRING_AGENT, callback_handler=None,
                 # Same session_id next time = the same conversation, loaded back from disk.
                 session_manager=FileSessionManager(session_id=session_id, storage_dir=str(SESSIONS)))


def run():
    """Show that a follow-up question works inside one session and fails in a new one."""
    with cv_client() as cv:
        tools = cv.list_tools_sync()
        dana = agent_for("dana", tools)
        dana("Who knows Kafka?")
        dana("Which of them lives in Haifa?")             # "them" works: same session
        omar = agent_for("omar", tools)
        omar("Which of them lives in Haifa?")             # new session: "them" means nothing
        print(f"Dana's session: {len(dana.messages)} messages. Omar's: {len(omar.messages)} messages.")
        print(f"Saved under {SESSIONS}/ — run again and Dana's conversation continues.")
        return len(dana.messages), len(omar.messages)


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    run()
