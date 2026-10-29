"""
STRANDS · STEP 2 · The first agent: a model, your CV tools, and a system prompt.

What it does
    Starts the CV MCP server, gives its tools to a Strands Agent, and asks one question.
    The agent decides which tools to call (find_candidates, get_cv, ...) and writes an answer.

How to run
    python run.py 02 "Which candidates know Kafka?" --framework strands
    python frameworks/strands/02_first_agent.py "Which candidates know Kafka and Docker? Check the evidence."

What you'll see
    One "calls" line per tool call (for example find_candidates({"skill": "kafka", ...})),
    then the final answer. With MODEL=fake the answer is a scripted stand-in, not a real model.

Same thing in LangChain: session1/02_first_agent.py
    LangChain   agent = create_agent(model, tools, system_prompt=...)   result = await agent.ainvoke({...})
    Strands     agent = Agent(model=..., tools=..., system_prompt=...)   result = agent("question")
Strands calls are normal (synchronous) calls: no async/await needed here.
"""
import pathlib
import sys

# Let Python find the shared "workshop" package that lives two folders up (the repo root).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from strands import Agent

from workshop.prompts import HIRING_AGENT
from workshop.strands_kit import READ_TOOLS, cv_client, show, strands_model


def run(question: str):
    """Build the agent, ask one question, print the tool calls and the answer, and return the result."""
    # `with` starts the CV MCP server and makes sure it is closed at the end, even after an error.
    with cv_client() as cv:
        tools = cv.list_tools_sync()             # ask the server for its tools (MCP tools/list)
        agent = Agent(
            model=strands_model(),               # MODEL in .env (MODEL=fake works offline)
            tools=tools + READ_TOOLS,            # MCP tools and a plain Python @tool, one list
            system_prompt=HIRING_AGENT,          # the rules: same prompt as the LangChain version
            callback_handler=None,               # don't stream tokens to the terminal; we print at the end
        )
        # Calling the agent like a function runs the whole loop:
        # the model thinks, asks for a tool, Strands runs it, the model reads the result... until an answer.
        result = agent(question)
        show(result, agent)
        return result


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    run(" ".join(sys.argv[1:]) or "Which candidates know Kafka and Docker? Check the evidence.")
