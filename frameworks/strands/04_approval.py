"""
STRANDS · STEP 4 · Limits and human approval before every write.
Same idea as session1/04_middleware_hitl.py (LangChain middleware), written with Strands hooks.

    python frameworks/strands/04_approval.py                 # you approve or reject in the terminal
    python frameworks/strands/04_approval.py --auto          # approve everything (tests)

LangChain has "middleware"; Strands has "hooks": functions that run at fixed moments of the loop
(before a model call, before a tool call, after a tool call…). Two hooks here:

    Limits     counts tool calls; past the budget it cancels them so the model must answer
    Approval   before a write tool, it INTERRUPTS: the agent stops and returns to your code,
               you ask a person, then you call the agent again with the answer to continue
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.hooks import BeforeInvocationEvent, BeforeToolCallEvent, HookProvider, HookRegistry

from workshop import config
from workshop.prompts import HIRING_AGENT
from workshop.strands_kit import ACTION_TOOLS, WRITES, cv_client, show, strands_model


class Limits(HookProvider):
    """At most 12 tool calls per question, and at most 6 full CVs (get_cv is the expensive one)."""

    def __init__(self, max_calls=12, max_cv=6):
        self.max_calls, self.max_cv = max_calls, max_cv

    def register_hooks(self, registry: HookRegistry, **kwargs):
        registry.add_callback(BeforeInvocationEvent, self.reset)        # a new question starts
        registry.add_callback(BeforeToolCallEvent, self.check)          # every tool call passes here

    def reset(self, event):
        self.calls, self.cvs = 0, 0

    def check(self, event: BeforeToolCallEvent):
        self.calls += 1
        self.cvs += event.tool_use["name"] == "get_cv"
        if self.calls > self.max_calls or self.cvs > self.max_cv:
            # cancel_tool = the tool does NOT run; the model reads this text instead and must answer.
            event.cancel_tool = "Budget reached: answer now with what you already have."


class Approval(HookProvider):
    """Before any write tool: stop and wait for a person."""

    def register_hooks(self, registry: HookRegistry, **kwargs):
        registry.add_callback(BeforeToolCallEvent, self.ask)

    def ask(self, event: BeforeToolCallEvent):
        name = event.tool_use["name"]
        if name not in WRITES:
            return                                                      # reads run freely
        # First time: this raises and the agent returns with stop_reason "interrupt".
        # After we resume with a response, the same line returns that response instead.
        answer = event.interrupt(f"approve-{name}", reason={"tool": name, "args": event.tool_use["input"]})
        if answer != "approve":
            event.cancel_tool = f"A person rejected {name}: {answer}"   # the reason goes back to the model


def ask_human(interrupt, auto: bool) -> str:
    print(f"\n⏸  The agent wants to run {interrupt.reason['tool']}({json.dumps(interrupt.reason['args'], ensure_ascii=False)})")
    if auto:
        return "approve"
    choice = input("   [a]pprove · [r]eject ? ").strip().lower()[:1]
    return "approve" if choice == "a" else (input("   why? ") or "Not approved.")


def run(question: str, auto=False):
    with cv_client() as cv:
        agent = Agent(model=strands_model(), tools=cv.list_tools_sync() + ACTION_TOOLS, system_prompt=HIRING_AGENT,
                      hooks=[Limits(), Approval()], callback_handler=None,
                      conversation_manager=SlidingWindowConversationManager(window_size=40))  # keep the last 40 messages
        result = agent(question)
        while result.stop_reason == "interrupt":                        # paused, waiting for a person
            responses = [{"interruptResponse": {"interruptId": i.id, "response": ask_human(i, auto)}}
                         for i in result.interrupts]
            result = agent(responses)                                   # resume exactly where it stopped
        show(result, agent)
        print(f"\nWhat really happened: see {config.OUTBOX}/")
        return result


if __name__ == "__main__":
    q = " ".join(a for a in sys.argv[1:] if not a.startswith("--")) or "Find a Kafka candidate and book an interview for the backend job."
    run(q, "--auto" in sys.argv)
