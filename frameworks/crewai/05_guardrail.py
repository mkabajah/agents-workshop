"""
CREWAI · STEP 5 · Your own rule: no booking without reading the CV first, plus an audit log.

What it does
    Two hooks you write yourself:
    before_tool_call  runs before every tool call; returning False blocks it (our rule)
    after_tool_call   runs after it; we use it to remember which CVs were read and to log every call

How to run
    python run.py 05 "Book an interview for a Kafka candidate for the backend job." --framework crewai
    python frameworks/crewai/05_guardrail.py "Book an interview for a Kafka candidate for the backend job."

What you'll see
    The tool calls, a "blocked schedule_interview" line if the agent tried to book before reading the CV,
    the answer, and "Blocked calls: N". Every call is logged in outbox/audit.jsonl.

Same thing in LangChain: session1/05_guardrail.py (@wrap_tool_call middleware).
A rule in code is stronger than a rule in the prompt: the prompt asks the model politely,
the hook makes it impossible.
"""
import json
import pathlib
import sys

# Let Python find the shared "workshop" package (repo root) and the step-2 script in this folder.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from crewai import Crew, Task
from crewai.hooks import after_tool_call, before_tool_call

from workshop import config
from workshop.crewai_kit import ACTION_TOOLS, short, show_tool_calls

# A file name starting with a digit can't be imported with a normal "import" line: use __import__.
first_agent = __import__("02_first_agent")
# CrewAI hooks don't see the whole conversation, so we keep our own record of the CVs the agent read.
READ_CVS, BLOCKED = set(), []


@before_tool_call
def evidence_first(ctx):
    """Block schedule_interview / send_email for a CV that was never opened with get_cv."""
    if short(ctx.tool_name) in ("schedule_interview", "send_email"):   # short(): drop the MCP server prefix
        cv = ctx.tool_input.get("cv", "")
        if cv not in READ_CVS:                                 # the rule, in code
            print(f"   ⛔ blocked {short(ctx.tool_name)}: {cv} was never read with get_cv")
            BLOCKED.append(cv)
            return False                                       # the tool does not run
    return None


@after_tool_call
def remember_and_log(ctx):
    """After each tool call: remember CVs read with get_cv, and append one line to the audit log."""
    if short(ctx.tool_name) == "get_cv":
        READ_CVS.add(ctx.tool_input.get("file_name", ""))
    config.OUTBOX.mkdir(exist_ok=True)
    with open(config.OUTBOX / "audit.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"framework": "crewai", "tool": short(ctx.tool_name), "args": ctx.tool_input}, ensure_ascii=False) + "\n")
    return None                                                # None = keep the tool's result as it is


def run(question: str):
    """Run the assistant with the write tools; the two hooks above enforce the rule and log every call."""
    show_tool_calls()
    # ACTION_TOOLS[1:] skips get_free_slots, which build_assistant() already adds.
    assistant = first_agent.build_assistant(extra_tools=ACTION_TOOLS[1:])
    task = Task(description=question, expected_output="What you did, and why, naming the CV files", agent=assistant)
    result = Crew(agents=[assistant], tasks=[task]).kickoff()
    print(f"\n🤖 answer {result}\nBlocked calls: {len(BLOCKED)} · audit log: {config.OUTBOX / 'audit.jsonl'}")
    return BLOCKED


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    run(" ".join(sys.argv[1:]) or "Book an interview for a Kafka candidate for the backend job.")
