"""
EXAMPLE 10 · CrewAI Flow: deterministic steps with a crew inside one step.        Framework: CrewAI

    pip install -r requirements-optional.txt
    python examples/10_crewai_flow.py backend-engineer
    python examples/10_crewai_flow.py backend-engineer --dry      # build only, no model call

    @start load_job ─▶ @listen run_crew ─▶ @router check ─┬─ "has_shortlist" ─▶ notify
                                                          └─ "empty"          ─▶ give_up
Flows are CrewAI's production shape: plain Python where the steps are known, a crew where
autonomy helps. Same lesson as LangGraph in session 1.

New to async/await, "async with" or decorators? Read PYTHON_NOTES.md (5 minutes).
"""
import pathlib
import sys

# Let Python find the shared "workshop" package in the repo root (PYTHON_NOTES.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "session2"))

from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel

from workshop import config

crew09 = __import__("09_crewai_crew")                             # reuse the sourcer + screener crew


class HiringState(BaseModel):
    job: str = "backend-engineer"
    post: str = ""
    report: str = ""


class HiringFlow(Flow[HiringState]):
    @start()
    def load_job(self):
        self.state.post = crew09.job_post(self.state.job)            # plain code: read the job post

    @listen(load_job)
    def run_crew(self):
        self.state.report = str(crew09.build().kickoff(inputs={"job": self.state.post}))   # the crew

    @router(run_crew)
    def check(self):
        return "has_shortlist" if ".md" in self.state.report or ".pdf" in self.state.report else "empty"

    @listen("has_shortlist")
    def notify(self):
        print("Shortlist ready for a human to review:\n", self.state.report[:1500])

    @listen("empty")
    def give_up(self):
        print("No candidate met the must-haves. Nothing to review.")


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    config.check()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flow = HiringFlow()
    if "--dry" in sys.argv:
        print("Built:", type(flow).__name__, "· steps:", sorted(flow._methods))
        sys.exit(0)
    flow.kickoff(inputs={"job": args[0] if args else "backend-engineer"})
