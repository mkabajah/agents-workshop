"""
CREWAI · STEP 3 · Memory: one conversation per recruiter.
Same as session1/03_memory.py (LangChain threads + a checkpointer).

    python frameworks/crewai/03_memory.py

CrewAI runs one crew per request and doesn't keep a chat thread for you. Its built-in memory
(Crew(memory=True)) is long-term, vector-based recall and needs an embedding model. For a chat
thread, the simple and honest way is: keep the conversation yourself and pass it into the next task.
That is all a "thread" is: the earlier messages, sent again.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from crewai import Crew, Task

first_agent = __import__("02_first_agent")          # reuse build_assistant() from step 2


class Thread:
    """One conversation: a list of (question, answer) pairs."""

    def __init__(self, name: str):
        self.name, self.turns = name, []

    def ask(self, question: str) -> str:
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
    dana, omar = Thread("dana"), Thread("omar")
    dana.ask("Who knows Kafka?")
    dana.ask("Which of them lives in Haifa?")             # "them" works: Dana's thread has the first answer
    omar.ask("Which of them lives in Haifa?")             # new thread: "them" means nothing
    print(f"Dana's thread: {len(dana.turns)} turns. Omar's: {len(omar.turns)} turn.")
    return dana, omar


if __name__ == "__main__":
    run()
