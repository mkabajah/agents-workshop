"""
EXAMPLE 09 · Strands Graph: a developer-drawn flow of agents, with a condition.     Framework: Strands

    pip install -r requirements-optional.txt
    python examples/09_strands_graph.py "Shortlist for the backend job"
    python examples/09_strands_graph.py --dry          # connect to MCP and build, no model call

    sourcer ──▶ screener ──(screener found evidence)──▶ writer
Same idea as session1/06 in LangGraph, in Strands' GraphBuilder. Nodes are Agents; edges can
carry a condition that reads the graph state.
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "session2"))

from strands import Agent
from strands.multiagent import GraphBuilder

from workshop import config, prompts

strands08 = __import__("08_strands_agents")                       # reuse strands_model() and rag_client()


def screener_found_evidence(state) -> bool:
    """Only write the summary when the screener produced a result."""
    result = state.results.get("screener")
    return bool(result and "missing" not in str(result.result).lower()[:40])


def build(tools):
    t = {x.tool_name: x for x in tools}
    model = strands08.strands_model()
    sourcer = Agent(name="sourcer", model=model, system_prompt=prompts.SOURCER,
                    tools=[t["find_candidates"], t["search_cvs"], t["list_jobs"]])
    screener = Agent(name="screener", model=model, system_prompt=prompts.SCREENER, tools=[t["get_cv"]])
    writer = Agent(name="writer", model=model, system_prompt="Write a 5-line shortlist summary for a hiring manager.")

    g = GraphBuilder()
    g.add_node(sourcer, "sourcer")
    g.add_node(screener, "screener")
    g.add_node(writer, "writer")
    g.add_edge("sourcer", "screener")
    g.add_edge("screener", "writer", condition=screener_found_evidence)
    g.set_entry_point("sourcer")
    g.set_max_node_executions(6)                                   # a budget, like every loop
    return g.build()


if __name__ == "__main__":
    config.check()
    ask = " ".join(a for a in sys.argv[1:] if not a.startswith("--")) or "Shortlist for the backend job"
    with strands08.rag_client() as rag:
        tools = rag.list_tools_sync()
        if "--dry" in sys.argv:
            os.environ.setdefault("AWS_REGION", "us-east-1")
            graph = build(tools)
            print("Built:", type(graph).__name__, "· nodes:", list(graph.nodes))
            sys.exit(0)
        result = build(tools)(ask)
        print("\nStatus:", result.status, "· order:", [n.node_id for n in result.execution_order])
