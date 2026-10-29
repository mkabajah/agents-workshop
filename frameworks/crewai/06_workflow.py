"""
CREWAI · STEP 6 · A workflow you can draw, with an agent in one step and a person in another.
Same process as session1/06_langgraph_screening.py (LangGraph StateGraph), written as a CrewAI Flow.

    python frameworks/crewai/06_workflow.py backend-engineer
    python frameworks/crewai/06_workflow.py backend-engineer --auto      # approve automatically (tests)

    @start load_job ─▶ @listen source ─▶ @router ─┬─ "has_matches" ─▶ screen (AGENT) ─▶ approve (PERSON) ─▶ save
                                                  └─ "nobody" ─▶ stop
A Flow is CrewAI's version of a graph: methods are the steps, decorators are the arrows,
self.state is the shared notebook (like LangGraph's state).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from crewai import Agent, Crew, Task
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel

from workshop.crewai_kit import RULES, crew_llm, cv_call, cv_server, save_shortlist
from workshop.jobs import must_haves

AUTO = "--auto" in sys.argv


class Screening(BaseModel):                       # the shared notebook every step reads and writes
    job: str = "backend-engineer"
    must_haves: list[str] = []
    matches: dict[str, list[str]] = {}
    shortlist: list[str] = []
    report: str = ""
    approved: bool = False


class ScreeningFlow(Flow[Screening]):
    @start()
    def load_job(self):                           # code
        post = next(j["text"] for j in cv_call("list_jobs") if j["job"] == self.state.job)
        self.state.must_haves = must_haves(post)

    @listen(load_job)
    def source(self):                             # code: one find_candidates per must-have
        for skill in self.state.must_haves:
            for hit in cv_call("find_candidates", skill=skill, city=""):
                self.state.matches.setdefault(hit["cv"], []).append(skill)
        self.state.shortlist = sorted(self.state.matches, key=lambda cv: -len(self.state.matches[cv]))[:3]
        print("must-haves:", self.state.must_haves, "· shortlist:", self.state.shortlist)

    @router(source)
    def anyone_found(self):                       # the fork in the road
        return "has_matches" if self.state.shortlist else "nobody"

    @listen("nobody")
    def stop(self):
        print("Nobody matches: stop here. No agent ran, no tokens spent.")

    @listen("has_matches")
    def screen(self):                             # AGENT: the one step that needs judgement
        screener = Agent(role="Hiring screener", goal="Check the evidence in each CV for every must-have",
                         backstory="You only trust what a CV shows, and you quote it." + RULES,
                         mcps=[cv_server()], llm=crew_llm(), max_iter=8)
        task = Task(description=f"Job: {self.state.job}. Must-haves: {', '.join(self.state.must_haves)}. "
                                f"Check the evidence in: {', '.join(self.state.shortlist)}.",
                    expected_output="A table: CV · must-have · evidence or 'missing' · one risk", agent=screener)
        self.state.report = str(Crew(agents=[screener], tasks=[task]).kickoff())

    @listen(screen)
    def approve(self):                            # PERSON
        print(f"\n⏸  Approve the shortlist for {self.state.job}? {self.state.shortlist}\nScreener report:\n{self.state.report[:1200]}")
        self.state.approved = AUTO or input("[approve / reject] ").strip() == "approve"

    @listen(approve)
    def save(self):                               # code
        if self.state.approved:
            save_shortlist.run(job=self.state.job, candidates=self.state.shortlist,
                               reasons=[f"has {', '.join(self.state.matches[c])}" for c in self.state.shortlist])
        print("Done · approved =", self.state.approved)


def run(job="backend-engineer"):
    flow = ScreeningFlow()
    flow.kickoff(inputs={"job": job})
    return flow.state


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(args[0] if args else "backend-engineer")
