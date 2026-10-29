"""The job posts (data/jobs/*.md) and the one rule we use to read their must-haves."""
import re

from workshop import config
from workshop.models import SKILLS


def job_post(job: str) -> str:
    """The text of one job post, e.g. job_post("backend-engineer")."""
    return (config.JOBS_DIR / f"{job}.md").read_text(encoding="utf-8")


def must_haves(post: str) -> list[str]:
    """The skills named under '## Must have': the first known skill on each line."""
    block = post.split("## Must have", 1)[-1].split("##", 1)[0].lower()
    out = []
    for line in block.splitlines():
        hits = sorted((m.start(), s) for s in SKILLS for m in [re.search(rf"(?<![a-z]){re.escape(s)}(?![a-z])", line)] if m)
        if hits:
            out.append(hits[0][1])
    return out
