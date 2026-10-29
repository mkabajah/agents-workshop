"""
CREWAI · STEP 3 · SHORT-TERM memory: one conversation per recruiter.

Short-term or long-term memory?
    SHORT-TERM (this step)  = the conversation itself: the messages of ONE thread. It lets the agent
                              understand "them" in a follow-up. A new thread starts empty.
                              CrewAI: we keep the turns ourselves and pass them in.
    LONG-TERM (examples/08) = facts that outlive a conversation, e.g. "Dana prefers Haifa".
                              Saved in a store, read and written through tools, shared by all of
                              Dana's threads. (CrewAI: Crew(memory=True), which needs an embedding model.)

What it does
    Dana asks two questions in the SAME conversation, so the follow-up "which of THEM lives in Haifa?"
    makes sense. Omar asks the same follow-up in a NEW conversation, so "them" means nothing.

How to run
    python run.py 03 --framework crewai
    python frameworks/crewai/03_memory.py

What you'll see
    Each question with a short answer, then "Dana's thread: 2 turns. Omar's: 1 turn."

Same thing in LangChain: session1/03_memory.py (a checkpointer + one thread_id per conversation).

How memory works here
    CrewAI runs one crew per request and doesn't keep a chat thread for you. Its built-in memory
    (Crew(memory=True)) is long-term, vector-based recall and needs an embedding model. For a chat
    thread, the simple and honest way is: keep the conversation yourself and pass it into the next task.
    That is all a "thread" is: the earlier messages, sent again. (It lives in memory only; it is gone
    when the script ends.)
"""
import pathlib
import sys

# Let Python find the shared "workshop" package (repo root) and the step-2 script in this folder.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from crewai import Crew, Task

# A file name starting with a digit can't be imported with a normal "import" line: use __import__.
first_agent = __import__("02_first_agent")          # reuse build_assistant() from step 2


class Thread:
    """One conversation: a list of (question, answer) pairs."""

    def __init__(self, name: str):
        """Start an empty conversation for one recruiter."""
        self.name, self.turns = name, []

    def ask(self, question: str) -> str:
        """Ask a question with the earlier turns pasted into the task, then remember the answer."""
        earlier = "\n".join(f"Recruiter: {q}\nYou: {a}" for q, a in self.turns)
        description = (f"Conversation so far:\n{earlier}\n\nNew question: {question}" if earlier else question)
        assistant = first_agent.build_assistant()
        task = Task(description=description, expected_output="A short answer naming the CV file for every claim",
                    agent=assistant)
        answer = str(Crew(agents=[assistant], tasks=[task]).kickoff())
        self.turns.append((question, answer))            # remember it for the next question
        print(f"[{self.name}] {question}\n   → {answer[:300]}\n")
        return answer


def run():
    """Show that a follow-up question works inside one thread and fails in a new one."""
    dana, omar = Thread("dana"), Thread("omar")
    dana.ask("Who knows Kafka?")
    dana.ask("Which of them lives in Haifa?")             # "them" works: Dana's thread has the first answer
    omar.ask("Which of them lives in Haifa?")             # new thread: "them" means nothing
    print(f"Dana's thread: {len(dana.turns)} turns. Omar's: {len(omar.turns)} turn.")
    return dana, omar


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    run()
