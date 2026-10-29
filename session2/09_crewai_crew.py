"""SESSION 2 · STEP 9 · The same team in CrewAI: roles, tasks and a crew.

What you learn
    A third way to describe the same team. CrewAI thinks in people and assignments: agents have a
    role, goal and backstory (together, their system prompt); tasks have a description and the
    expected output; a crew runs the tasks under a process (sequential here: source, then screen).
    MCP: the `mcps=[...]` field on an Agent connects the CV server to that agent.
    CrewAI is synchronous: crew.kickoff(...) just returns the result.

How to run
    pip install -r requirements-optional.txt
    python session2/09_crewai_crew.py backend-engineer
    python session2/09_crewai_crew.py backend-engineer --dry      # build only, no model call

What you'll see
    CrewAI's own log of each agent's thoughts and tool calls (verbose=True), then the final table:
    candidate, must-haves met with evidence, gap or risk, and the top 3.

Model: CREW_LLM (a LiteLLM name, e.g. "anthropic/claude-sonnet-4-6", "ollama/qwen2.5:7b",
"groq/openai/gpt-oss-20b"), or derived from MODEL ("anthropic:x" -> "anthropic/x"), or MODEL=fake.
"""
import json
import pathlib
import sys

# Let Python find the shared "workshop" package in the repo root (docs/python-notes.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from crewai import Agent, Crew, Process, Task

from workshop import config
from workshop.crewai_kit import RULES, crew_llm, cv_server
from workshop.jobs import job_post



def build():
    """Two agents and two tasks, run one after the other; the second task sees the first's result."""
    rag = cv_server()                                                              # the CV MCP server
    llm = crew_llm()

    sourcer = Agent(role="Technical sourcer", goal="Find every candidate who meets the job's must-haves",
                    backstory="You never miss a candidate; you search once per skill." + RULES,
                    mcps=[rag], llm=llm)
    screener = Agent(role="Hiring screener", goal="Check the evidence in each CV and pick the best three",
                     backstory="You only trust what a CV shows, and you quote it." + RULES,
                     mcps=[rag], llm=llm)

    # {job} is filled in at kickoff(inputs={"job": ...}). expected_output tells the agent when it is done.
    source = Task(description="Job post:\n{job}\n\nFind every candidate who meets each must-have. "
                              "Use find_candidates once per skill.",
                  expected_output="CV file names, each with the must-haves it meets", agent=sourcer)
    screen = Task(description="Read the CVs of the candidates found (get_cv). Check each must-have with a short quote.",
                  expected_output="A table: candidate · must-haves met with evidence · gap or risk; then the top 3 and why",
                  agent=screener, context=[source])               # context: gets the sourcer's output
    return Crew(agents=[sourcer, screener], tasks=[source, screen], process=Process.sequential, verbose=True)


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    config.check()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    job = args[0] if args else "backend-engineer"
    crew = build()
    if "--dry" in sys.argv:
        print(json.dumps({"agents": [a.role for a in crew.agents], "tasks": len(crew.tasks), "llm": str(getattr(crew.agents[0].llm, "model", crew.agents[0].llm))}))
        sys.exit(0)
    print(crew.kickoff(inputs={"job": job_post(job)}))
