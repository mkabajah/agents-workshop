"""
CREWAI · STEP 6 · A workflow you can draw, with an agent in one step and a person in another.

What it does
    Screens candidates for one job as a CrewAI Flow. Only ONE step needs a model:

    @start load_job ─▶ @listen source ─▶ @router ─┬─ "has_matches" ─▶ screen (AGENT) ─▶ approve (PERSON) ─▶ save
                                                  └─ "nobody" ─▶ stop

How to run
    python run.py 06 backend-engineer --framework crewai
    python frameworks/crewai/06_workflow.py backend-engineer
    python frameworks/crewai/06_workflow.py backend-engineer --auto      # approve automatically (tests)

What you'll see
    The must-haves and the shortlist (for backend-engineer: cand-006, cand-018, cand-007), the
    screener's report, a pause for your approval, and "Done · approved = True". CrewAI also prints
    boxes showing the flow's progress. An approved shortlist is written to outbox/shortlists.jsonl.

Same thing in LangChain: session1/06_langgraph_screening.py (a LangGraph StateGraph with interrupt()).

How a Flow works
    A Flow is CrewAI's version of a graph: methods are the steps, decorators are the arrows,
    self.state is the shared notebook (like LangGraph's state).
    @start()            runs first
    @listen(step)       runs after that step finishes
    @router(step)       runs after that step and returns a label; @listen("label") picks the branch
"""
import pathlib
import sys

# Let Python find the shared "workshop" package that lives two folders up (the repo root).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from crewai import Agent, Crew, Task
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel

from workshop.crewai_kit import RULES, crew_llm, cv_call, cv_server, save_shortlist
from workshop.jobs import must_haves

AUTO = "--auto" in sys.argv


class Screening(BaseModel):                       # the shared notebook every step reads and writes
    """The flow's state: every field a step needs to read or fill in, with its type and default."""
    job: str = "backend-engineer"
    must_haves: list[str] = []
    matches: dict[str, list[str]] = {}            # CV file name -> must-haves it has
    shortlist: list[str] = []
    report: str = ""
    approved: bool = False


class ScreeningFlow(Flow[Screening]):
    """The screening process: code steps, one agent step, one person step."""

    @start()
    def load_job(self):                           # code
        """Read the job post (through the MCP server) and pick out its must-haves."""
        # cv_call() calls one MCP tool directly: no agent, no model, no tokens.
        post = next(j["text"] for j in cv_call("list_jobs") if j["job"] == self.state.job)
        self.state.must_haves = must_haves(post)

    @listen(load_job)
    def source(self):                             # code: one find_candidates per must-have
        """Find who has each must-have and keep the top 3 by how many they have."""
        for skill in self.state.must_haves:
            for hit in cv_call("find_candidates", skill=skill, city=""):
                self.state.matches.setdefault(hit["cv"], []).append(skill)
        self.state.shortlist = sorted(self.state.matches, key=lambda cv: -len(self.state.matches[cv]))[:3]
        print("must-haves:", self.state.must_haves, "· shortlist:", self.state.shortlist)

    @router(source)
    def anyone_found(self):                       # the fork in the road
        """Return the label of the next branch: "has_matches" or "nobody"."""
        # The labels must differ from the method names, or CrewAI refuses the flow (it would loop).
        return "has_matches" if self.state.shortlist else "nobody"

    @listen("nobody")
    def stop(self):
        """Nobody matched: end here without running any agent."""
        print("Nobody matches: stop here. No agent ran, no tokens spent.")

    @listen("has_matches")
    def screen(self):                             # AGENT: the one step that needs judgement
        """Let a screener agent check the evidence in the shortlisted CVs."""
        screener = Agent(role="Hiring screener", goal="Check the evidence in each CV for every must-have",
                         backstory="You only trust what a CV shows, and you quote it." + RULES,
                         mcps=[cv_server()], llm=crew_llm(), max_iter=8)
        task = Task(description=f"Job: {self.state.job}. Must-haves: {', '.join(self.state.must_haves)}. "
                                f"Check the evidence in: {', '.join(self.state.shortlist)}.",
                    expected_output="A table: CV · must-have · evidence or 'missing' · one risk", agent=screener)
        self.state.report = str(Crew(agents=[screener], tasks=[task]).kickoff())

    @listen(screen)
    def approve(self):                            # PERSON
        """Show the shortlist and the report, and ask a person to approve."""
        print(f"\n⏸  Approve the shortlist for {self.state.job}? {self.state.shortlist}\nScreener report:\n{self.state.report[:1200]}")
        self.state.approved = AUTO or input("[approve / reject] ").strip() == "approve"

    @listen(approve)
    def save(self):                               # code
        """Write the shortlist to outbox/ only if it was approved."""
        if self.state.approved:
            # A CrewAI @tool is an object, not a plain function: .run(...) calls it directly.
            save_shortlist.run(job=self.state.job, candidates=self.state.shortlist,
                               reasons=[f"has {', '.join(self.state.matches[c])}" for c in self.state.shortlist])
        print("Done · approved =", self.state.approved)


def run(job="backend-engineer"):
    """Run the flow for one job and return its final state."""
    flow = ScreeningFlow()
    flow.kickoff(inputs={"job": job})             # inputs fill the matching fields of the state
    return flow.state


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(args[0] if args else "backend-engineer")
