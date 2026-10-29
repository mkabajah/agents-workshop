"""
EXAMPLE 11 · Deploy-ready for AWS: a Strands agent behind Bedrock AgentCore Runtime.     AWS

    pip install bedrock-agentcore strands-agents
    python examples/11_agentcore_strands.py          # serves http://localhost:8080 like AgentCore does
    curl -X POST localhost:8080/invocations -H 'Content-Type: application/json' \\
         -d '{"prompt": "Who knows Kafka?"}'

AgentCore Runtime runs your agent in an isolated microVM per session and calls ONE entrypoint
with the request payload. The same few lines wrap any framework: see 12 for LangGraph.
On AWS the RAG server is not a local process: host it as an MCP server (AgentCore Runtime or
Gateway) and set RAG_MCP_URL + MCP_TOKEN. Deploying is the AWS session's topic.

New to async/await, "async with" or decorators? Read PYTHON_NOTES.md (5 minutes).
"""
import pathlib
import sys

# Let Python find the shared "workshop" package in the repo root (PYTHON_NOTES.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "session2"))

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent

from workshop import prompts

strands08 = __import__("08_strands_agents")
app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict) -> dict:
    """AgentCore calls this once per request. Validate input before it reaches the agent."""
    prompt = str(payload.get("prompt", ""))[:2000]
    if not prompt:
        return {"error": "Send {'prompt': '...'}"}
    with strands08.rag_client() as rag:                         # one MCP session per request
        agent = Agent(model=strands08.strands_model(), tools=rag.list_tools_sync(),
                      system_prompt=prompts.HIRING_AGENT, callback_handler=None)
        result = agent(prompt)
    return {"answer": str(result)}


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    app.run()                                                     # port 8080: /invocations and /ping
