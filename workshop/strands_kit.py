"""
STRANDS HELPERS · everything the Strands versions share (frameworks/strands/, session2/08, examples):
the model, the CV tools over MCP, and the write tools.

    strands_model()   MODEL → a Strands model (anthropic:…, openai:…, ollama:…, fake, or bedrock)
    cv_client()       an MCPClient for the CV server (bundled, or yours via RAG_REPO)
    ACTION_TOOLS      get_free_slots, schedule_interview, send_email, save_shortlist as Strands tools

Strands is synchronous by default: agent("question") just returns the result. (Under the hood it
runs an async loop for you, so you don't need async/await in these scripts.)

The @tool docstrings below are read by the model as tool descriptions, like in workshop/actions.py.
"""
import json
import os
import sys
import uuid

from mcp import StdioServerParameters, stdio_client
from strands import tool
from strands.models import Model
from strands.tools.mcp import MCPClient

from workshop import actions, config, mcp_tools
from workshop.models import decide


# ---------------------------------------------------------------- the model
def strands_model(name: str | None = None):
    """The Strands model object for MODEL (or `name`). Each provider has its own class in strands.models;
    we import it only when needed, so you only install the provider you use."""
    # "anthropic:claude-x".partition(":") → ("anthropic", ":", "claude-x")
    provider, _, model_id = (name or config.MODEL).partition(":")
    if provider == "fake":
        return FakeStrandsModel()
    if provider == "anthropic":
        from strands.models.anthropic import AnthropicModel
        return AnthropicModel(model_id=model_id, max_tokens=2000)
    if provider == "openai":
        from strands.models.openai import OpenAIModel
        args = {"base_url": os.environ["OPENAI_BASE_URL"]} if os.getenv("OPENAI_BASE_URL") else {}
        return OpenAIModel(model_id=model_id, client_args=args)
    if provider == "ollama":
        from strands.models.ollama import OllamaModel
        return OllamaModel(host="http://localhost:11434", model_id=model_id)
    return None                                   # None = Strands' default model on Amazon Bedrock


class FakeStrandsModel(Model):
    """The same rule-following stand-in as workshop/models.py, speaking Strands' streaming format.
    Not an LLM: it lets every Strands script run with no key and no network."""

    def __init__(self):
        """No settings to load; Strands only asks for a config dict."""
        self.config = {"model_id": "fake"}

    def update_config(self, **model_config):
        """Part of the Model interface Strands expects; we just store the values."""
        self.config.update(model_config)

    def get_config(self):
        """Part of the Model interface: return the settings."""
        return self.config

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        """Typed answers need a real model; the fake says so clearly instead of guessing."""
        raise NotImplementedError("The fake model has no structured output; use a real model.")
        yield  # pragma: no cover  (makes this an async generator, as the interface requires)

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        """One model call. Strands expects a stream of small events (start, content, stop), so we
        "yield" them one by one: either a tool request or a text answer, as decide() chooses."""
        # The question = the last user message that has text (tool results are user messages too).
        last = max((i for i, m in enumerate(messages) if m["role"] == "user" and any("text" in c for c in m["content"])), default=0)
        question = " ".join(c["text"] for c in messages[last]["content"] if "text" in c)
        done, outputs = [], []
        for m in messages[last + 1:]:
            for c in m["content"]:
                if "toolUse" in c:
                    done.append((c["toolUse"]["name"], c["toolUse"]["input"]))
                if "toolResult" in c:
                    outputs += [b.get("text", "") for b in c["toolResult"]["content"]]
        specs = tool_specs or []
        names = [s["name"] for s in specs]
        args = {s["name"]: list(s["inputSchema"]["json"].get("properties", {})) for s in specs}
        kind, name, a = decide(question, names, args, done, " ".join(outputs))
        # The events below are the same shape a real model (e.g. Bedrock) streams back to Strands.
        yield {"messageStart": {"role": "assistant"}}
        if kind == "call":
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"tool_{uuid.uuid4().hex[:8]}", "name": name}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(a)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockDelta": {"delta": {"text": name}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}


# ---------------------------------------------------------------- the CV tools, over MCP
def cv_client() -> MCPClient:
    """An MCPClient for the CV server. Use it with `with`, which starts the server once and stops it after:

        with cv_client() as cv:
            tools = cv.list_tools_sync()        # MCP tools/list, as Strands tools
    """
    config.check()
    cv = mcp_tools.server_config()["cv"]
    if cv["transport"] != "stdio":
        sys.exit("The Strands versions use the local server; unset RAG_MCP_URL.")
    # MCPClient wants a function that opens the connection (it calls it when the `with` block starts).
    return MCPClient(lambda: stdio_client(StdioServerParameters(
        command=cv["command"], args=cv["args"], env=cv["env"], cwd=cv["cwd"])))


# ---------------------------------------------------------------- the write tools (same mocks → outbox/)
@tool
def get_free_slots() -> list[str]:
    """The interview slots that are still free this week."""
    return actions.get_free_slots.invoke({})


@tool
def schedule_interview(cv: str, job: str, slot: str) -> str:
    """Book an interview for one candidate (CV file name) for a job, in a free slot."""
    return actions.schedule_interview.invoke({"cv": cv, "job": job, "slot": slot})


@tool
def send_email(cv: str, subject: str, body: str) -> str:
    """Email one candidate (by CV file name). The address is looked up by HR, never by you."""
    return actions.send_email.invoke({"cv": cv, "subject": subject, "body": body})


@tool
def save_shortlist(job: str, candidates: list[str], reasons: list[str]) -> str:
    """Save the shortlist for a job: CV file names and one reason each."""
    return actions.save_shortlist.invoke({"job": job, "candidates": candidates, "reasons": reasons})


# Ready-made tool lists; WRITES = the tools that need a person's OK in 04_approval.py.
READ_TOOLS = [get_free_slots]
ACTION_TOOLS = [get_free_slots, schedule_interview, send_email, save_shortlist]
WRITES = {"schedule_interview", "send_email", "save_shortlist"}


def show(result, agent=None):
    """Print the run like the LangChain trace: every tool call, then the answer."""
    msgs = agent.messages if agent is not None else []
    for m in msgs:
        for c in m["content"]:
            if "toolUse" in c:
                print(f"🤖 calls  {c['toolUse']['name']}({json.dumps(c['toolUse']['input'], ensure_ascii=False)})")
    print(f"\n🤖 answer {str(result).strip()[:1200]}")
