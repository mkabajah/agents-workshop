"""
EXAMPLE 09 · Strands Graph: a developer-drawn flow of agents, with a condition.     Framework: Strands

    pip install -r requirements-optional.txt
    python examples/09_strands_graph.py "Shortlist for the backend job"
    python examples/09_strands_graph.py --dry          # connect to MCP and build, no model call

    sourcer ──▶ screener ──(only if the screener found evidence)──▶ writer

What it shows: in Strands, a Graph's nodes are whole Agents (each with its own prompt and tools), and you
draw the edges. An edge can carry a condition: a plain Python function that reads the graph state and
returns True or False. Same idea as session1/06 in LangGraph, but every node is an agent.

When you'd use it: a pipeline of specialists where the ORDER is fixed but each step needs judgement.
What you'll see: the run's status and the order the nodes ran in.
For a fixed process with code steps, see frameworks/strands/06_workflow.py.
"""
import os
import pathlib
import sys

# Let Python find the shared "workshop" package in the repo root (docs/python-notes.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from strands import Agent
from strands.multiagent import GraphBuilder

from workshop import config, prompts
from workshop.strands_kit import cv_client, strands_model



def screener_found_evidence(state) -> bool:
    """Edge condition: only go on to the writer when the screener produced a usable result.
    state.results holds each finished node's output, by node name."""
    result = state.results.get("screener")
    return bool(result and "missing" not in str(result.result).lower()[:40])


def build(tools):
    """Three agents, each with only the tools it needs, wired into a graph with a step budget."""
    t = {x.tool_name: x for x in tools}
    model = strands_model()
    sourcer = Agent(name="sourcer", model=model, system_prompt=prompts.SOURCER,
                    tools=[t["find_candidates"], t["search_cvs"], t["list_jobs"]])
    screener = Agent(name="screener", model=model, system_prompt=prompts.SCREENER, tools=[t["get_cv"]])
    writer = Agent(name="writer", model=model, system_prompt="Write a 5-line shortlist summary for a hiring manager.")

    g = GraphBuilder()
    g.add_node(sourcer, "sourcer")
    g.add_node(screener, "screener")
    g.add_node(writer, "writer")
    g.add_edge("sourcer", "screener")
    g.add_edge("screener", "writer", condition=screener_found_evidence)   # fires only if this is True
    g.set_entry_point("sourcer")
    g.set_max_node_executions(6)                                   # a budget, like every loop
    return g.build()


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    config.check()
    ask = " ".join(a for a in sys.argv[1:] if not a.startswith("--")) or "Shortlist for the backend job"
    with cv_client() as rag:                       # start the CV MCP server; it stops after the block
        tools = rag.list_tools_sync()
        if "--dry" in sys.argv:
            os.environ.setdefault("AWS_REGION", "us-east-1")   # Strands' default model wants a region
            graph = build(tools)
            print("Built:", type(graph).__name__, "· nodes:", list(graph.nodes))
            sys.exit(0)
        result = build(tools)(ask)
        print("\nStatus:", result.status, "· order:", [n.node_id for n in result.execution_order])
