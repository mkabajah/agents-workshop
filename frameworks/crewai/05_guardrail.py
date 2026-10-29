"""
CREWAI · STEP 5 · Your own rule: no booking without reading the CV first, plus an audit log.
Same as session1/05_guardrail.py (LangChain @wrap_tool_call), written as CrewAI tool hooks.

    python frameworks/crewai/05_guardrail.py "Book an interview for a Kafka candidate for the backend job."

before_tool_call  runs before every tool call; returning False blocks it
after_tool_call   runs after it; we use it to remember which CVs were read and to log every call
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from crewai import Crew, Task
from crewai.hooks import after_tool_call, before_tool_call

from workshop import config
from workshop.crewai_kit import ACTION_TOOLS, short, show_tool_calls

first_agent = __import__("02_first_agent")
READ_CVS, BLOCKED = set(), []


@before_tool_call
def evidence_first(ctx):
    if short(ctx.tool_name) in ("schedule_interview", "send_email"):
        cv = ctx.tool_input.get("cv", "")
        if cv not in READ_CVS:                                 # the rule, in code
            print(f"   ⛔ blocked {short(ctx.tool_name)}: {cv} was never read with get_cv")
            BLOCKED.append(cv)
            return False                                       # the tool does not run
    return None


@after_tool_call
def remember_and_log(ctx):
    if short(ctx.tool_name) == "get_cv":
        READ_CVS.add(ctx.tool_input.get("file_name", ""))
    config.OUTBOX.mkdir(exist_ok=True)
    with open(config.OUTBOX / "audit.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"framework": "crewai", "tool": short(ctx.tool_name), "args": ctx.tool_input}, ensure_ascii=False) + "\n")
    return None                                                # None = keep the tool's result as it is


def run(question: str):
    show_tool_calls()
    assistant = first_agent.build_assistant(extra_tools=ACTION_TOOLS[1:])
    task = Task(description=question, expected_output="What you did, and why, naming the CV files", agent=assistant)
    result = Crew(agents=[assistant], tasks=[task]).kickoff()
    print(f"\n🤖 answer {result}\nBlocked calls: {len(BLOCKED)} · audit log: {config.OUTBOX / 'audit.jsonl'}")
    return BLOCKED


if __name__ == "__main__":
    run(" ".join(sys.argv[1:]) or "Book an interview for a Kafka candidate for the backend job.")
