"""
EXAMPLE 10 · Deploy-ready for AWS: a Strands agent behind Bedrock AgentCore Runtime.     AWS

    pip install bedrock-agentcore strands-agents
    python examples/10_agentcore_strands.py          # serves http://localhost:8080 like AgentCore does
    curl -X POST localhost:8080/invocations -H 'Content-Type: application/json' \\
         -d '{"prompt": "Who knows Kafka?"}'

What it shows: AgentCore Runtime runs your agent in an isolated microVM per session and calls ONE
function, your "entrypoint", with the request's JSON payload. BedrockAgentCoreApp gives you the same
web server locally: POST /invocations calls the entrypoint, GET /ping answers health checks.

    POST /invocations {"prompt": ...} ──▶ invoke(payload) ──▶ Strands Agent + CV tools ──▶ {"answer": ...}

When you'd use it: when the agent must run as a service other apps call, not as a script.
What you'll see: the server starts on port 8080; the curl command returns {"answer": "..."}.
The same few lines wrap any framework: see 11 for LangGraph.
On AWS the CV server is not a local process: host it as an MCP server (AgentCore Runtime or Gateway)
and set RAG_MCP_URL + MCP_TOKEN. Deploying is the AWS session's topic.
"""
import pathlib
import sys

# Let Python find the shared "workshop" package in the repo root (docs/python-notes.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent

from workshop import prompts
from workshop.strands_kit import cv_client, strands_model

# The app object is the web server; @app.entrypoint below marks the one function it calls per request.
app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict) -> dict:
    """AgentCore calls this once per request. Validate input before it reaches the agent."""
    # Treat the payload as untrusted input: take only the field we need and cap its length.
    prompt = str(payload.get("prompt", ""))[:2000]
    if not prompt:
        return {"error": "Send {'prompt': '...'}"}
    with cv_client() as rag:                         # one MCP session per request
        agent = Agent(model=strands_model(), tools=rag.list_tools_sync(),
                      system_prompt=prompts.HIRING_AGENT, callback_handler=None)
        result = agent(prompt)
    return {"answer": str(result)}


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    app.run()                                                     # port 8080: /invocations and /ping
