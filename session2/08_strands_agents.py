"""
SESSION 2 · STEP 8 · The same team in Strands Agents (AWS, open source).

    pip install -r requirements-optional.txt
    python session2/08_strands_agents.py "Shortlist 3 candidates for the backend job"          # agents as tools
    python session2/08_strands_agents.py --swarm "Shortlist 3 candidates for the backend job"  # a swarm
    python session2/08_strands_agents.py --dry                                                  # build only, no model call

Strands is "model-driven": an Agent is a model + tools + a prompt, and the model plans the steps.
Same MCP server, different client: strands.tools.mcp.MCPClient.

Model: taken from MODEL (anthropic:…, openai:…, ollama:…) or MODEL=bedrock for Strands' default.

New to async/await, "async with" or decorators? Read PYTHON_NOTES.md (5 minutes).
"""
import os
import pathlib
import sys

# Let Python find the shared "workshop" package in the repo root (PYTHON_NOTES.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.multiagent import Swarm
from strands.tools.mcp import MCPClient

from workshop import config, mcp_tools, prompts


def strands_model():
    provider, _, name = config.MODEL.partition(":")
    if provider == "anthropic":
        from strands.models.anthropic import AnthropicModel
        return AnthropicModel(model_id=name, max_tokens=2000)
    if provider == "openai":
        from strands.models.openai import OpenAIModel
        args = {"base_url": os.environ["OPENAI_BASE_URL"]} if os.getenv("OPENAI_BASE_URL") else {}
        return OpenAIModel(model_id=name, client_args=args)
    if provider == "ollama":
        from strands.models.ollama import OllamaModel
        return OllamaModel(host="http://localhost:11434", model_id=name)
    if provider == "fake":
        sys.exit("MODEL=fake is a LangChain stand-in; Strands needs a real model (try MODEL=ollama:qwen2.5:7b).")
    return None                                                     # None = Strands' default (Amazon Bedrock)


def rag_client() -> MCPClient:
    cv = mcp_tools.server_config()["cv"]
    return MCPClient(lambda: stdio_client(StdioServerParameters(
        command=cv["command"], args=cv["args"], env=cv["env"], cwd=cv["cwd"])))


def build(rag_tools, swarm=False):
    by_name = {t.tool_name: t for t in rag_tools}
    model = strands_model()
    sourcer = Agent(name="sourcer", description="Finds EVERY candidate who has given must-haves.",
                    model=model, system_prompt=prompts.SOURCER,
                    tools=[by_name["find_candidates"], by_name["search_cvs"]])
    screener = Agent(name="screener", description="Checks the evidence in specific CVs and finds risks.",
                     model=model, system_prompt=prompts.SCREENER,
                     tools=[by_name["get_cv"], by_name["search_cvs"]])
    if swarm:
        # Hand over control: each agent gets a handoff_to_agent tool and a shared context.
        return Swarm([sourcer, screener], entry_point=sourcer, max_handoffs=6, max_iterations=10,
                     repetitive_handoff_detection_window=4, repetitive_handoff_min_unique_agents=2)
    # Delegate and return: specialists are just tools of the lead.
    return Agent(model=model, system_prompt=prompts.LEAD.replace("- coordinator", "- (no coordinator here)"),
                 tools=[by_name["list_jobs"], sourcer, screener])


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    config.check()
    use_swarm, dry = "--swarm" in sys.argv, "--dry" in sys.argv
    ask = " ".join(a for a in sys.argv[1:] if not a.startswith("--")) or "Shortlist 3 candidates for the backend job."
    with rag_client() as rag:                                     # opens one MCP session for the run
        tools = rag.list_tools_sync()
        print("MCP tools:", [t.tool_name for t in tools])
        if dry:
            os.environ.setdefault("AWS_REGION", "us-east-1")
            team = build(tools, use_swarm)
            print("Built:", type(team).__name__)
            sys.exit(0)
        team = build(tools, use_swarm)
        result = team(ask)
        if use_swarm:
            print("\nStatus:", result.status, "· path:", [n.node_id for n in result.node_history])
