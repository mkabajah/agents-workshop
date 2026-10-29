"""
CREWAI · STEP 2 · The first agent: a role, your CV tools, and one task.

What it does
    Builds one CrewAI agent connected to the CV MCP server, gives it one task (your question),
    and runs a crew of one. The agent decides which tools to call and writes the answer.

How to run
    python run.py 02 "Which candidates know Kafka?" --framework crewai
    python frameworks/crewai/02_first_agent.py "Which candidates know Kafka and Docker? Check the evidence."

What you'll see
    One "calls" line per tool call (for example find_candidates({"skill": "kafka", ...})),
    then the final answer. With MODEL=fake the answer is a scripted stand-in, not a real model.

Same thing in LangChain: session1/02_first_agent.py
    LangChain   create_agent(model, tools, system_prompt)            → agent.ainvoke({"messages": [...]})
    CrewAI      Agent(role, goal, backstory, mcps, tools, llm)
                + Task(description, expected_output) + Crew(...)    → crew.kickoff()
CrewAI thinks in people and assignments: the "system prompt" is split into role, goal and backstory,
and every question becomes a Task with the answer you expect. crew.kickoff() is a normal
(synchronous) call: no async/await needed here.

Tool names: CrewAI names MCP tools "<server>_<tool>", so find_candidates may show up with a long
prefix built from the server's command. workshop/crewai_kit.short() turns it back into the short name.
"""
import pathlib
import sys

# Let Python find the shared "workshop" package that lives two folders up (the repo root).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from crewai import Agent, Crew, Task

from workshop.crewai_kit import RULES, crew_llm, cv_server, get_free_slots, show_tool_calls


def build_assistant(extra_tools=(), max_iter=12) -> Agent:
    """The recruiting assistant used by steps 2 to 5. Pass extra_tools to give it write tools."""
    return Agent(
        # role + goal + backstory together are the agent's system prompt, written as a job description.
        role="Recruiting assistant",
        goal="Answer the hiring team's questions about candidates, with evidence from their CVs",
        backstory="You are careful: every claim comes from a CV you read." + RULES,
        mcps=[cv_server()],                  # the CV MCP server: CrewAI starts it and lists its tools
        tools=[get_free_slots, *extra_tools],  # plain Python tools next to the MCP tools
        llm=crew_llm(),                      # MODEL / CREW_LLM in .env (MODEL=fake works offline)
        max_iter=max_iter,                   # the step budget: never loop forever
        verbose=False)


def run(question: str):
    """Ask one question with a crew of one agent and print the answer."""
    show_tool_calls()                        # print each tool call as it happens
    assistant = build_assistant()
    # A Task is one assignment. expected_output tells the agent what a finished answer looks like;
    # CrewAI adds it to the prompt, so it shapes the answer.
    task = Task(description=question,
                expected_output="A short answer that names the CV file for every claim",
                agent=assistant)
    result = Crew(agents=[assistant], tasks=[task]).kickoff()   # runs the loop and returns the answer
    print(f"\n🤖 answer {result}")
    return result


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    run(" ".join(sys.argv[1:]) or "Which candidates know Kafka and Docker? Check the evidence.")
