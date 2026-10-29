"""
CREWAI · STEP 2 · The first agent: a role, your CV tools, and one task.
Same as session1/02_first_agent.py (LangChain), written with CrewAI.

    python frameworks/crewai/02_first_agent.py "Which candidates know Kafka and Docker? Check the evidence."

Compare with LangChain:
    LangChain   create_agent(model, tools, system_prompt)            → agent.ainvoke({"messages": [...]})
    CrewAI      Agent(role, goal, backstory, mcps, tools, llm)
                + Task(description, expected_output) + Crew(...)    → crew.kickoff()
CrewAI thinks in people and assignments: the "system prompt" is split into role, goal and backstory,
and every question becomes a Task with the answer you expect.
"""
import pathlib
import sys

# Let Python find the shared "workshop" package that lives two folders up (the repo root).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from crewai import Agent, Crew, Task

from workshop.crewai_kit import RULES, crew_llm, cv_server, get_free_slots, show_tool_calls


def build_assistant(extra_tools=(), max_iter=12) -> Agent:
    return Agent(
        role="Recruiting assistant",
        goal="Answer the hiring team's questions about candidates, with evidence from their CVs",
        backstory="You are careful: every claim comes from a CV you read." + RULES,
        mcps=[cv_server()],                  # the CV MCP server: CrewAI starts it and lists its tools
        tools=[get_free_slots, *extra_tools],  # plain Python tools next to the MCP tools
        llm=crew_llm(),                      # MODEL / CREW_LLM in .env (MODEL=fake works offline)
        max_iter=max_iter,                   # the step budget: never loop forever
        verbose=False)


def run(question: str):
    show_tool_calls()                        # print each tool call as it happens
    assistant = build_assistant()
    task = Task(description=question,
                expected_output="A short answer that names the CV file for every claim",
                agent=assistant)
    result = Crew(agents=[assistant], tasks=[task]).kickoff()   # runs the loop and returns the answer
    print(f"\n🤖 answer {result}")
    return result


if __name__ == "__main__":
    run(" ".join(sys.argv[1:]) or "Which candidates know Kafka and Docker? Check the evidence.")
