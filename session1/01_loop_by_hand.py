"""SESSION 1 · STEP 1 · The agent loop by hand: no agent framework, just a chat model and MCP.

What you learn
    What an "agent" really is: a loop. The model never runs anything itself; it only ASKS for a
    tool call, and YOUR code runs the tool and hands the result back. Every framework you meet later
    (LangChain, Strands, CrewAI, the OpenAI and Claude SDKs) is this loop plus extras.

How to run
    python session1/01_loop_by_hand.py "Who knows Kafka?"
    MODEL=fake python session1/01_loop_by_hand.py "Who knows Kafka?"     # offline, no API key

What happens, step by step
  1. We start the CV MCP server and ask it for its tools      (MCP: tools/list)
  2. We send the question + the tool descriptions to the model
  3. The model either answers, or ASKS for a tool call         (it never runs anything itself)
  4. OUR code runs the tool through MCP                        (MCP: tools/call)
  5. The result goes back into the message list. Repeat until the model stops asking.

What you'll see
    "Tools from your MCP server: [...]", then a trace: the question (user), each tool request
    (calls), each tool result, and the final answer.

Comes before: nothing. This is the foundation. Step 2 replaces this loop with one call.

New to async/await, "async with" or decorators? Read docs/python-notes.md (5 minutes).
"""
import asyncio
import pathlib
import sys

# Let Python find the shared "workshop" package in the repo root (docs/python-notes.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from workshop import config, mcp_tools
from workshop.models import get_model
from workshop.prompts import HIRING_AGENT
from workshop.show import banner, trace

MAX_STEPS = 8                                            # never `while True`: an agent needs a budget


async def run(question: str):
    """Run the agent loop for one question and return the full list of messages.

    It is async because talking to the MCP server and to the model means waiting;
    every wait is marked with await (docs/python-notes.md §1)."""
    config.check()
    # The same server settings Claude Desktop uses: which program to start, with which arguments.
    cv = mcp_tools.server_config()["cv"]
    server = StdioServerParameters(command=cv["command"], args=cv["args"], env=cv["env"], cwd=cv["cwd"])

    # Start the CV MCP server as a child process and talk to it through its stdin/stdout ("stdio").
    # "async with" closes the session and stops the server at the end, even after an error.
    async with stdio_client(server) as (read, write), ClientSession(read, write) as mcp:
        await mcp.initialize()                                             # the MCP handshake
        listed = await mcp.list_tools()                                    # ① MCP tools/list
        # Turn each MCP tool into the JSON "function" format that chat models understand.
        schemas = [{"type": "function", "function": {
            "name": t.name, "description": t.description, "parameters": t.inputSchema}} for t in listed.tools]
        print("Tools from your MCP server:", [t.name for t in listed.tools])

        model = get_model().bind_tools(schemas)                            # ② the model sees the schemas
        # The message list IS the agent's working memory: rules first, then the question.
        # Every model call re-reads the whole list, so it grows with each step.
        messages = [SystemMessage(HIRING_AGENT), HumanMessage(question)]

        for step in range(1, MAX_STEPS + 1):
            reply = await model.ainvoke(messages)                          # ③ answer, or ask for tools
            messages.append(reply)
            if not reply.tool_calls:                                       # no tool call → done
                break
            # A tool call looks like {"name": "find_candidates", "args": {"skill": "kafka"}, "id": "call_1"}.
            # The model may ask for several at once; we run each one.
            for call in reply.tool_calls:
                result = await mcp.call_tool(call["name"], call["args"])  # ④ OUR code acts (tools/call)
                # The answer comes back as a list of content blocks; we join their text.
                text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
                # tool_call_id links this result to the request it answers; models require it.
                messages.append(ToolMessage(text, tool_call_id=call["id"], name=call["name"]))   # ⑤ observe
        else:                                                              # for…else: runs only if we never hit "break"
            print(f"Stopped after {MAX_STEPS} steps: step limit.")

    banner(f"The loop ran {step} model call(s)")
    trace(messages)
    return messages


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    # asyncio.run() starts the async function and waits until it is finished.
    asyncio.run(run(" ".join(sys.argv[1:]) or "Who knows Kafka?"))
