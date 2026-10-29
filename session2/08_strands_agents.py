"""
SESSION 2 · STEP 8 · The same team in Strands Agents (AWS, open source).

    pip install -r requirements-optional.txt
    python session2/08_strands_agents.py "Shortlist 3 candidates for the backend job"          # agents as tools
    python session2/08_strands_agents.py --swarm "Shortlist 3 candidates for the backend job"  # a swarm
    python session2/08_strands_agents.py --dry                                                  # build only, no model call

Strands is "model-driven": an Agent is a model + tools + a prompt, and the model plans the steps.
Same MCP server, different client: strands.tools.mcp.MCPClient.

Model: taken from MODEL (anthropic:…, openai:…, ollama:…) or MODEL=bedrock for Strands' default.

"""
import os
import pathlib
import sys

# Let Python find the shared "workshop" package in the repo root (docs/python-notes.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from strands import Agent
from strands.multiagent import Swarm

from workshop import config, prompts
from workshop.strands_kit import cv_client, strands_model


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
    with cv_client() as rag:                                     # opens one MCP session for the run
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
