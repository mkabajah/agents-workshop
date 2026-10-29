"""
CREWAI · STEP 4 · Limits and human approval before every write.
Same idea as session1/04_middleware_hitl.py (LangChain middleware), written with CrewAI hooks.

    python frameworks/crewai/04_approval.py               # you approve or reject in the terminal
    python frameworks/crewai/04_approval.py --auto        # approve everything (tests)

Limits:   Agent(max_iter=12) is the step budget.
Approval: a @before_tool_call hook runs before EVERY tool call. For a write tool it asks a person;
          returning False blocks the call, so the tool never runs and the agent is told it was blocked.
          (CrewAI asks in the same process: the run waits at input(). For approvals that can wait
          hours, LangGraph's interrupt + checkpointer is the stronger tool.)
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from crewai import Crew, Task
from crewai.hooks import before_tool_call

from workshop import config
from workshop.crewai_kit import ACTION_TOOLS, WRITES, short, show_tool_calls

first_agent = __import__("02_first_agent")
AUTO = "--auto" in sys.argv
DECISIONS = []                                              # what the person decided, for the summary


@before_tool_call
def approval(ctx):
    name = short(ctx.tool_name)
    if name not in WRITES:
        return None                                         # reads run freely
    print(f"\n⏸  The agent wants to run {name}({json.dumps(ctx.tool_input, ensure_ascii=False)})")
    ok = AUTO or input("   [a]pprove · [r]eject ? ").strip().lower().startswith("a")
    DECISIONS.append((name, "approve" if ok else "reject"))
    return None if ok else False                            # False = blocked: the tool does not run


def run(question: str):
    show_tool_calls()
    assistant = first_agent.build_assistant(extra_tools=ACTION_TOOLS[1:], max_iter=12)
    task = Task(description=question, expected_output="What you did, and why, naming the CV files", agent=assistant)
    result = Crew(agents=[assistant], tasks=[task]).kickoff()
    print(f"\n🤖 answer {result}\nDecisions: {DECISIONS}\nWhat really happened: see {config.OUTBOX}/")
    return DECISIONS


if __name__ == "__main__":
    run(" ".join(a for a in sys.argv[1:] if not a.startswith("--")) or "Find a Kafka candidate and book an interview for the backend job.")
