"""
STRANDS · STEP 4 · Limits and human approval before every write.

What it does
    The agent can now ACT (book interviews, send emails, save shortlists). Two safety rules:
    a budget on tool calls, and a person must approve every write before it happens.

How to run
    python run.py 04 --framework strands
    python frameworks/strands/04_approval.py                 # you approve or reject in the terminal
    python frameworks/strands/04_approval.py --auto          # approve everything (used by the tests)

What you'll see
    The agent searches, then stops at a line like "The agent wants to run schedule_interview(...)".
    Approve: the booking lands in outbox/interviews.jsonl. Reject: the tool never runs and the model
    is told why.

Same thing in LangChain: session1/04_middleware_hitl.py (middleware + interrupt + Command(resume=...)).

Hooks, in one paragraph
    LangChain has "middleware"; Strands has "hooks": functions that Strands calls at fixed moments of the
    loop (before a question starts, before each tool call, after each tool call, ...). You group hooks in
    a class that extends HookProvider; its register_hooks() method tells the HookRegistry "call this
    function when that event happens". Two hook classes here:

    Limits     counts tool calls; past the budget it cancels them so the model must answer
    Approval   before a write tool, it INTERRUPTS: the agent stops and returns to your code,
               you ask a person, then you call the agent again with the answer to continue
"""
import json
import pathlib
import sys

# Let Python find the shared "workshop" package that lives two folders up (the repo root).
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
        """Remember the two budgets."""
        self.max_calls, self.max_cv = max_calls, max_cv

    def register_hooks(self, registry: HookRegistry, **kwargs):
        """Tell Strands which of our functions to call, and when."""
        registry.add_callback(BeforeInvocationEvent, self.reset)        # a new question starts
        registry.add_callback(BeforeToolCallEvent, self.check)          # every tool call passes here

    def reset(self, event):
        """New question: start counting from zero."""
        self.calls, self.cvs = 0, 0

    def check(self, event: BeforeToolCallEvent):
        """Count this call; past a budget, cancel it instead of running it."""
        self.calls += 1
        self.cvs += event.tool_use["name"] == "get_cv"                  # True counts as 1, False as 0
        if self.calls > self.max_calls or self.cvs > self.max_cv:
            # cancel_tool = the tool does NOT run; the model reads this text instead and must answer.
            event.cancel_tool = "Budget reached: answer now with what you already have."


class Approval(HookProvider):
    """Before any write tool: stop and wait for a person."""

    def register_hooks(self, registry: HookRegistry, **kwargs):
        """Run ask() before every tool call."""
        registry.add_callback(BeforeToolCallEvent, self.ask)

    def ask(self, event: BeforeToolCallEvent):
        """Let reads through; pause the agent on writes and act on the person's answer."""
        name = event.tool_use["name"]
        if name not in WRITES:
            return                                                      # reads run freely
        # First time: this raises and the agent returns with stop_reason "interrupt".
        # After we resume with a response, the same line returns that response instead.
        # The name must be unique per tool; the reason is what we show the person.
        answer = event.interrupt(f"approve-{name}", reason={"tool": name, "args": event.tool_use["input"]})
        if answer != "approve":
            event.cancel_tool = f"A person rejected {name}: {answer}"   # the reason goes back to the model


def ask_human(interrupt, auto: bool) -> str:
    """Show the pending action in the terminal and return "approve" or the person's reason to reject."""
    print(f"\n⏸  The agent wants to run {interrupt.reason['tool']}({json.dumps(interrupt.reason['args'], ensure_ascii=False)})")
    if auto:
        return "approve"
    choice = input("   [a]pprove · [r]eject ? ").strip().lower()[:1]
    return "approve" if choice == "a" else (input("   why? ") or "Not approved.")


def run(question: str, auto=False):
    """Run the agent with both hooks; every time it pauses, ask a person and resume."""
    with cv_client() as cv:
        agent = Agent(model=strands_model(), tools=cv.list_tools_sync() + ACTION_TOOLS, system_prompt=HIRING_AGENT,
                      hooks=[Limits(), Approval()], callback_handler=None,
                      conversation_manager=SlidingWindowConversationManager(window_size=40))  # keep the last 40 messages
        result = agent(question)
        # An interrupt does not end the run: the agent is paused and remembers exactly where it was.
        while result.stop_reason == "interrupt":                        # paused, waiting for a person
            # One answer per pending interrupt, matched by its id.
            responses = [{"interruptResponse": {"interruptId": i.id, "response": ask_human(i, auto)}}
                         for i in result.interrupts]
            result = agent(responses)                                   # resume exactly where it stopped
        show(result, agent)
        print(f"\nWhat really happened: see {config.OUTBOX}/")
        return result


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    q = " ".join(a for a in sys.argv[1:] if not a.startswith("--")) or "Find a Kafka candidate and book an interview for the backend job."
    run(q, "--auto" in sys.argv)
