"""
STRANDS · STEP 5 · Your own rule: no booking without reading the CV first, plus an audit log.
Same as session1/05_guardrail.py (LangChain @wrap_tool_call), written as two Strands hooks.

    python frameworks/strands/05_guardrail.py "Book an interview for a Kafka candidate for the backend job."
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from strands import Agent
from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent, HookProvider, HookRegistry

from workshop import config
from workshop.prompts import HIRING_AGENT
from workshop.strands_kit import ACTION_TOOLS, cv_client, show, strands_model


class EvidenceFirst(HookProvider):
    """Block schedule_interview / send_email unless the agent read that CV with get_cv."""

    def register_hooks(self, registry: HookRegistry, **kwargs):
        registry.add_callback(BeforeToolCallEvent, self.check)

    def check(self, event: BeforeToolCallEvent):
        call = event.tool_use
        if call["name"] not in ("schedule_interview", "send_email"):
            return
        cv = call["input"].get("cv", "")
        already_read = any(c.get("toolUse", {}).get("name") == "get_cv" and cv and cv in json.dumps(c["toolUse"]["input"])
                           for m in event.agent.messages for c in m["content"])
        if not already_read:
            event.cancel_tool = f"Blocked: read the CV first with get_cv('{cv}'), then explain why this candidate fits."


class AuditLog(HookProvider):
    """Write every tool call to outbox/audit.jsonl: which tool, which arguments, how long."""

    def register_hooks(self, registry: HookRegistry, **kwargs):
        registry.add_callback(BeforeToolCallEvent, self.start)
        registry.add_callback(AfterToolCallEvent, self.stop)

    def start(self, event):
        self.t0 = time.perf_counter()

    def stop(self, event: AfterToolCallEvent):
        config.OUTBOX.mkdir(exist_ok=True)
        with open(config.OUTBOX / "audit.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"framework": "strands", "tool": event.tool_use["name"], "args": event.tool_use["input"],
                                "ms": round((time.perf_counter() - self.t0) * 1000)}, ensure_ascii=False) + "\n")


def run(question: str):
    with cv_client() as cv:
        agent = Agent(model=strands_model(), tools=cv.list_tools_sync() + ACTION_TOOLS, system_prompt=HIRING_AGENT,
                      hooks=[AuditLog(), EvidenceFirst()], callback_handler=None)
        result = agent(question)
        show(result, agent)
        blocked = [c["toolResult"] for m in agent.messages for c in m["content"]
                   if "toolResult" in c and "Blocked:" in json.dumps(c["toolResult"])]
        print(f"\nBlocked calls: {len(blocked)} · audit log: {config.OUTBOX / 'audit.jsonl'}")
        return agent, blocked


if __name__ == "__main__":
    run(" ".join(sys.argv[1:]) or "Book an interview for a Kafka candidate for the backend job.")
