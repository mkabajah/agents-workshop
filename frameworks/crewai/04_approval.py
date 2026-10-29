"""
CREWAI · STEP 4 · Limits and human approval before every write.

What it does
    The agent can now ACT (book interviews, send emails, save shortlists). Two safety rules:
    a step budget, and a person must approve every write before it happens.

How to run
    python run.py 04 --framework crewai
    python frameworks/crewai/04_approval.py               # you approve or reject in the terminal
    python frameworks/crewai/04_approval.py --auto        # approve everything (used by the tests)

What you'll see
    The agent searches, then stops at "The agent wants to run schedule_interview(...)".
    Approve: the booking lands in outbox/interviews.jsonl. Reject: the tool never runs.
    At the end, a list of your decisions.

Same thing in LangChain: session1/04_middleware_hitl.py (middleware + interrupt + Command(resume=...)).

How it works in CrewAI
    Limits:   Agent(max_iter=12) is the step budget.
    Approval: a @before_tool_call hook runs before EVERY tool call, for every agent in this process.
              For a write tool it asks a person; returning False blocks the call, so the tool never
              runs and the agent is told it was blocked. Returning None lets the call run.
              (CrewAI asks in the same process: the run waits at input(). For approvals that can wait
              hours, LangGraph's interrupt + checkpointer is the stronger tool.)
"""
import json
import pathlib
import sys

# Let Python find the shared "workshop" package (repo root) and the step-2 script in this folder.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from crewai import Crew, Task
from crewai.hooks import before_tool_call

from workshop import config
from workshop.crewai_kit import ACTION_TOOLS, WRITES, short, show_tool_calls

# A file name starting with a digit can't be imported with a normal "import" line: use __import__.
first_agent = __import__("02_first_agent")
AUTO = "--auto" in sys.argv
DECISIONS = []                                              # what the person decided, for the summary


# The decorator registers this function as soon as the file is loaded: from then on CrewAI calls it
# before every tool call. ctx describes the call: ctx.tool_name and ctx.tool_input (the arguments).
@before_tool_call
def approval(ctx):
    """Ask a person before any write tool; return False to block it, None to let it run."""
    name = short(ctx.tool_name)                             # drop the MCP server prefix from the name
    if name not in WRITES:
        return None                                         # reads run freely
    print(f"\n⏸  The agent wants to run {name}({json.dumps(ctx.tool_input, ensure_ascii=False)})")
    ok = AUTO or input("   [a]pprove · [r]eject ? ").strip().lower().startswith("a")
    DECISIONS.append((name, "approve" if ok else "reject"))
    return None if ok else False                            # False = blocked: the tool does not run


def run(question: str):
    """Run the assistant with the write tools; the approval hook guards every write."""
    show_tool_calls()
    # ACTION_TOOLS[1:] skips get_free_slots, which build_assistant() already adds.
    assistant = first_agent.build_assistant(extra_tools=ACTION_TOOLS[1:], max_iter=12)
    task = Task(description=question, expected_output="What you did, and why, naming the CV files", agent=assistant)
    result = Crew(agents=[assistant], tasks=[task]).kickoff()
    print(f"\n🤖 answer {result}\nDecisions: {DECISIONS}\nWhat really happened: see {config.OUTBOX}/")
    return DECISIONS


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    run(" ".join(a for a in sys.argv[1:] if not a.startswith("--")) or "Find a Kafka candidate and book an interview for the backend job.")
