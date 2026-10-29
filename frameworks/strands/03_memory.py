"""
STRANDS · STEP 3 · Memory: one session per conversation.
Same as session1/03_memory.py (LangChain threads + a checkpointer).

    python frameworks/strands/03_memory.py

In Strands an Agent object remembers its own conversation (agent.messages). To keep a
conversation across restarts, give it a session manager: here a FileSessionManager that saves
each session as files under outbox/sessions/. One session_id = one conversation (= a LangGraph thread).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from strands import Agent
from strands.session.file_session_manager import FileSessionManager

from workshop import config
from workshop.prompts import HIRING_AGENT
from workshop.strands_kit import cv_client, strands_model

SESSIONS = config.OUTBOX / "sessions"


def agent_for(session_id: str, tools) -> Agent:
    """One agent per conversation. The session manager loads earlier messages and saves new ones."""
    return Agent(model=strands_model(), tools=tools, system_prompt=HIRING_AGENT, callback_handler=None,
                 session_manager=FileSessionManager(session_id=session_id, storage_dir=str(SESSIONS)))


def run():
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


if __name__ == "__main__":
    run()
