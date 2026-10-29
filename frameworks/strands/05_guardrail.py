"""
STRANDS · STEP 5 · Your own rule: no booking without reading the CV first, plus an audit log.

What it does
    Adds two hooks you write yourself:
    EvidenceFirst   blocks schedule_interview / send_email unless the agent read that CV with get_cv
    AuditLog        writes every tool call (which tool, which arguments, how long) to outbox/audit.jsonl

How to run
    python run.py 05 "Book an interview for a Kafka candidate for the backend job." --framework strands
    python frameworks/strands/05_guardrail.py "Book an interview for a Kafka candidate for the backend job."

What you'll see
    The tool calls, the answer, and "Blocked calls: N". A blocked call never ran: the model got the
    "Blocked: ..." message instead, so it can read the CV first and try again.

Same thing in LangChain: session1/05_guardrail.py (@wrap_tool_call middleware).
A rule in code is stronger than a rule in the prompt: the prompt asks the model politely,
the hook makes it impossible.
"""
import json
import pathlib
import sys
import time

# Let Python find the shared "workshop" package that lives two folders up (the repo root).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from strands import Agent
from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent, HookProvider, HookRegistry

from workshop import config
from workshop.prompts import HIRING_AGENT
from workshop.strands_kit import ACTION_TOOLS, cv_client, show, strands_model


class EvidenceFirst(HookProvider):
    """Block schedule_interview / send_email unless the agent read that CV with get_cv."""

    def register_hooks(self, registry: HookRegistry, **kwargs):
        """Run check() before every tool call."""
        registry.add_callback(BeforeToolCallEvent, self.check)

    def check(self, event: BeforeToolCallEvent):
        """Look back through the conversation for a get_cv call on the same CV; cancel the call if none."""
        call = event.tool_use                                     # {"name": ..., "input": {...}}
        if call["name"] not in ("schedule_interview", "send_email"):
            return
        cv = call["input"].get("cv", "")
        # event.agent.messages is the whole conversation so far; tool calls are "toolUse" blocks in it.
        already_read = any(c.get("toolUse", {}).get("name") == "get_cv" and cv and cv in json.dumps(c["toolUse"]["input"])
                           for m in event.agent.messages for c in m["content"])
        if not already_read:
            # cancel_tool: the tool does not run, and the model reads this text as the tool's result.
            event.cancel_tool = f"Blocked: read the CV first with get_cv('{cv}'), then explain why this candidate fits."


class AuditLog(HookProvider):
    """Write every tool call to outbox/audit.jsonl: which tool, which arguments, how long."""

    def register_hooks(self, registry: HookRegistry, **kwargs):
        """Start a timer before each tool call and write a log line after it."""
        registry.add_callback(BeforeToolCallEvent, self.start)
        registry.add_callback(AfterToolCallEvent, self.stop)

    def start(self, event):
        """Remember when the tool call started."""
        self.t0 = time.perf_counter()

    def stop(self, event: AfterToolCallEvent):
        """Append one JSON line per tool call (JSON Lines: one record per line, easy to read back)."""
        config.OUTBOX.mkdir(exist_ok=True)
        with open(config.OUTBOX / "audit.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"framework": "strands", "tool": event.tool_use["name"], "args": event.tool_use["input"],
                                "ms": round((time.perf_counter() - self.t0) * 1000)}, ensure_ascii=False) + "\n")


def run(question: str):
    """Run the agent with both hooks and report how many calls the rule blocked."""
    with cv_client() as cv:
        # Hooks run in list order: the audit timer starts before the rule decides.
        agent = Agent(model=strands_model(), tools=cv.list_tools_sync() + ACTION_TOOLS, system_prompt=HIRING_AGENT,
                      hooks=[AuditLog(), EvidenceFirst()], callback_handler=None)
        result = agent(question)
        show(result, agent)
        # Find the tool results that carry our "Blocked:" message.
        blocked = [c["toolResult"] for m in agent.messages for c in m["content"]
                   if "toolResult" in c and "Blocked:" in json.dumps(c["toolResult"])]
        print(f"\nBlocked calls: {len(blocked)} · audit log: {config.OUTBOX / 'audit.jsonl'}")
        return agent, blocked


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    run(" ".join(sys.argv[1:]) or "Book an interview for a Kafka candidate for the backend job.")
