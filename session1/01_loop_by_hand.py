"""
SESSION 1 · STEP 1 · The agent loop by hand: no agent framework, just a chat model and MCP.

    python session1/01_loop_by_hand.py "Who knows Kafka?"

What you see:
  1. We start YOUR RAG MCP server and ask it for its tools      (MCP: tools/list)
  2. We send the question + the tool schemas to the model
  3. The model either answers, or ASKS for a tool call           (it never runs anything itself)
  4. OUR code runs the tool through MCP                         (MCP: tools/call)
  5. The result goes back into the messages. Repeat until the model stops asking.

Every agent framework (LangChain, Strands, CrewAI, the OpenAI and Claude SDKs) is this loop
plus memory, limits, streaming, hooks and tracing.
"""
import asyncio
import pathlib
import sys

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
    config.check()
    cv = mcp_tools.server_config()["cv"]
    server = StdioServerParameters(command=cv["command"], args=cv["args"], env=cv["env"], cwd=cv["cwd"])

    async with stdio_client(server) as (read, write), ClientSession(read, write) as mcp:
        await mcp.initialize()
        listed = await mcp.list_tools()                                    # ① MCP tools/list
        schemas = [{"type": "function", "function": {
            "name": t.name, "description": t.description, "parameters": t.inputSchema}} for t in listed.tools]
        print("Tools from your MCP server:", [t.name for t in listed.tools])

        model = get_model().bind_tools(schemas)                            # ② the model sees the schemas
        messages = [SystemMessage(HIRING_AGENT), HumanMessage(question)]

        for step in range(1, MAX_STEPS + 1):
            reply = await model.ainvoke(messages)                          # ③ answer, or ask for tools
            messages.append(reply)
            if not reply.tool_calls:                                       # no tool call → done
                break
            for call in reply.tool_calls:
                result = await mcp.call_tool(call["name"], call["args"])  # ④ OUR code acts (tools/call)
                text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
                messages.append(ToolMessage(text, tool_call_id=call["id"], name=call["name"]))   # ⑤ observe
        else:
            print(f"Stopped after {MAX_STEPS} steps: step limit.")

    banner(f"The loop ran {step} model call(s)")
    trace(messages)
    return messages


if __name__ == "__main__":
    asyncio.run(run(" ".join(sys.argv[1:]) or "Who knows Kafka?"))
