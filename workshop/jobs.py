"""
JOBS · the job posts (data/jobs/*.md) and the one rule we use to read their must-haves.

Used by the three workflow versions (session1/06, frameworks/strands/06, frameworks/crewai/06) and by
session2/09, so they all read a job post the same way and pick the same shortlist.
"""
import re

from workshop import config
from workshop.models import SKILLS


def job_post(job: str) -> str:
    """The text of one job post, e.g. job_post("backend-engineer")."""
    return (config.JOBS_DIR / f"{job}.md").read_text(encoding="utf-8")


def must_haves(post: str) -> list[str]:
    """The skills named under '## Must have': the first known skill on each line."""
    # Keep only the text between "## Must have" and the next "##" heading.
    block = post.split("## Must have", 1)[-1].split("##", 1)[0].lower()
    out = []
    for line in block.splitlines():
        # (?<![a-z]) and (?![a-z]) = whole words only, so "go" doesn't match inside "google".
        hits = sorted((m.start(), s) for s in SKILLS for m in [re.search(rf"(?<![a-z]){re.escape(s)}(?![a-z])", line)] if m)
        if hits:
            out.append(hits[0][1])
    return out
