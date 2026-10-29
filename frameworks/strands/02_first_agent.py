"""
STRANDS · STEP 2 · The first agent: a model, your CV tools, and a system prompt.
Same as session1/02_first_agent.py (LangChain), written with Strands Agents.

    python frameworks/strands/02_first_agent.py "Which candidates know Kafka and Docker? Check the evidence."

Compare with LangChain:
    LangChain   agent = create_agent(model, tools, system_prompt=...)   result = await agent.ainvoke({...})
    Strands     agent = Agent(model=..., tools=..., system_prompt=...)   result = agent("question")
"""
import pathlib
import sys

# Let Python find the shared "workshop" package that lives two folders up (the repo root).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from strands import Agent

from workshop.prompts import HIRING_AGENT
from workshop.strands_kit import READ_TOOLS, cv_client, show, strands_model


def run(question: str):
    # `with` starts the CV MCP server and makes sure it is closed at the end, even after an error.
    with cv_client() as cv:
        tools = cv.list_tools_sync()             # ask the server for its tools (MCP tools/list)
        agent = Agent(
            model=strands_model(),               # MODEL in .env (MODEL=fake works offline)
            tools=tools + READ_TOOLS,            # MCP tools and a plain Python @tool, one list
            system_prompt=HIRING_AGENT,          # the rules: same prompt as the LangChain version
            callback_handler=None,               # don't stream tokens to the terminal; we print at the end
        )
        result = agent(question)                 # runs the whole loop: think → tool → think → answer
        show(result, agent)
        return result


if __name__ == "__main__":
    run(" ".join(sys.argv[1:]) or "Which candidates know Kafka and Docker? Check the evidence.")
